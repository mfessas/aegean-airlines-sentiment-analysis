"""Recalculate the quantitative visuals in Final Presentation.pptx (slides 8–15).
Save under Python Code/Presentation. Run with Python 3.10+.
No inference, API calls or changes to input files. Outputs PNG, SVG and CSV.
"""
from pathlib import Path
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
HF = ROOT / 'Data/Output/Hugging Face/df_final_absa.csv'
GEMINI = ROOT / 'Data/Output/LLM/df_final_absa_gemini.csv'
OUTPUT = ROOT / 'Presentation'
EUROPE = ['Aegean Airlines','Turkish Airlines','Air France','Lufthansa']
REGIONAL = ['Aegean Airlines','Bangkok Air','Porter Airlines','Virgin Australia']
AIRLINES = EUROPE + REGIONAL[1:]
AE = AIRLINES[0]
FEATURES = 'cancellation delay boarding baggage customer_service cabin_crew ground_staff complaint_resolution seat_comfort food_beverage cabin_cleanliness entertainment_wifi booking_experience check_in airport_experience value_for_money pricing_fees flight_disruption'.split()
LABELS = {'delay':'Delays','pricing_fees':'Pricing & fees','food_beverage':'Food & beverage','entertainment_wifi':'Entertainment / Wi-Fi'}
SENTIMENTS = ['positive','negative','neutral','mixed']
NAVY, WHITE, GOLD, GREEN, RED, GRAY = '#102747','#F4F7FC','#D5B36B','#49B88B','#E66A70','#9AA9BC'
plt.rcParams.update({'figure.facecolor':NAVY,'axes.facecolor':NAVY,'text.color':WHITE,'axes.labelcolor':WHITE,'xtick.color':WHITE,'ytick.color':WHITE,'axes.edgecolor':NAVY,'font.size':12,'savefig.facecolor':NAVY,'axes.titleweight':'bold','svg.fonttype':'none'})

def label(f):
    return LABELS.get(f, f.replace('_',' ').capitalize())

def flag(s):
    return s.astype('string').str.strip().str.lower().isin(['1','1.0','true','yes','y'])

def load(path, columns=None):
    d = pd.read_csv(path, low_memory=False, dtype={'record_key':'string'}, usecols=columns)
    d['record_key'] = d.record_key.str.strip()
    if d.record_key.isna().any() or d.record_key.eq('').any() or d.record_key.duplicated().any():
        raise ValueError(f'Invalid or duplicate record_key in {path}')
    return d

def prepare(hf_path, gem_path, exclude=False):
    """Use HF metadata and Gemini predictions, matching one-to-one."""
    h = load(hf_path)
    cols = ['record_key','gemini_overall_sentiment'] + [f+s for f in FEATURES for s in ('_llm','_sentiment_llm')]
    g = load(gem_path, cols)
    d = h.drop(columns=[c for c in cols[1:] if c in h]).merge(g,on='record_key',validate='one_to_one')
    matched = len(d)
    d['review_date'] = pd.to_datetime(d.review_date, errors='coerce', utc=True).dt.tz_localize(None)
    d = d[d.review_date.dt.year.between(2024,2026)].copy()
    aliases = {'aegean':'Aegean Airlines','aegean_airlines':'Aegean Airlines','bangkok_airways':'Bangkok Air','bangkok airways':'Bangkok Air', **{a.lower().replace(' ','_'):a for a in AIRLINES}, **{a.lower():a for a in AIRLINES}}
    d['airline'] = d.airline.astype('string').str.strip().str.lower().map(aliases)
    d = d[d.airline.isin(AIRLINES)].copy()
    if exclude:
        d = d[~(flag(d.long_haul)|flag(d.first_class))].copy()
    if d.empty: raise ValueError('No eligible matched reviews')
    d['period'] = np.where(d.review_date.dt.year.eq(2026),'2026','2024–2025')
    for c in ['gemini_overall_sentiment']+[f+'_sentiment_llm' for f in FEATURES]:
        d[c] = d[c].astype('string').str.strip().str.lower()
    # Do not silently treat failed or unclassified predictions as neutral.
    if not d.gemini_overall_sentiment.isin(SENTIMENTS).all():
        raise ValueError('Unrecognized Gemini overall labels in matched sample; inspect input CSV.')
    print(f'Matched before filters: {matched:,}; presentation sample: {len(d):,}')
    if not exclude and len(d)!=25568: warnings.warn('Sample differs from final presentation (25,568). Check input versions and scope.')
    return d

def save(fig, name, data):
    """Export a high-resolution raster, editable vector and underlying values."""
    fig.savefig(OUTPUT/f'{name}.png',dpi=300,bbox_inches='tight',pad_inches=.25)
    fig.savefig(OUTPUT/f'{name}.svg',bbox_inches='tight',pad_inches=.25)
    data.to_csv(OUTPUT/f'{name}.csv',index=False,encoding='utf-8-sig')
    plt.close(fig)

def canvas(title, n=1, size=(14,6)):
    fig, axes = plt.subplots(1,n,figsize=size,squeeze=False,layout='constrained')
    fig.suptitle(title,fontsize=20,color=GOLD,fontweight='bold')
    for ax in axes.flat:
        ax.spines[:].set_visible(False); ax.tick_params(length=0)
    return fig, axes[0]

def horizontal(ax, names, values, color=GOLD, fmt=lambda v:f'{v:,.0f}'):
    bars=ax.barh(names,values,color=color,height=.65); ax.invert_yaxis()
    ax.bar_label(bars,labels=[fmt(v) for v in values],padding=5,color=WHITE)
    ax.margins(x=.22); ax.set_xticks([])

def eda(d):
    fig, axes=canvas(f'Exploring the sample | {len(d):,} matched reviews',3,(19,6))
    source_alias={'tripadvisor':'TripAdvisor','trustpilot':'Trustpilot','kayak':'Kayak','esky':'eSky','flightreport':'Flight-Report','airlineratings':'AirlineRatings'}
    source=d.source.astype('string').str.strip(); source=source.str.lower().str.replace(r'[^a-z]','',regex=True).map(source_alias).fillna(source)
    lang=d.original_language.astype('string').str.strip(); lang=lang[lang.notna() & ~lang.str.lower().isin(['','unknown','undetermined','nan','none'])]
    tables=[]
    for ax,title,series in zip(axes,['Reviews by source','Top 7 original languages','Reviews by airline'],[source.value_counts(),lang.value_counts().head(7),d.airline.value_counts()]):
        ax.set_title(title); horizontal(ax,series.index,series.values)
        tables.append(series.rename_axis('category').reset_index(name='reviews').assign(panel=title))
    print(f'Known languages: {lang.nunique()}'); save(fig,'slide_08_sample',pd.concat(tables,ignore_index=True))

def validation():
    # Slide 9 notes explicitly state these are presenter-supplied historical scores.
    t=pd.DataFrame({'metric':['Overall accuracy','Overall Macro F1','Aspect detection F1'],'Hugging Face':[.738,.435,.310],'Gemini':[.964,.486,.821]})
    t['gap_pp']=100*(t.Gemini-t['Hugging Face'])
    fig,(ax,)=canvas('Model validation | Hugging Face vs Gemini')
    x=np.arange(len(t)); w=.34
    for offset,model,color in [(-w/2,'Hugging Face',GRAY),(w/2,'Gemini',GREEN)]:
        b=ax.bar(x+offset,t[model],w,label=model,color=color); ax.bar_label(b,fmt='%.3f',padding=4,color=WHITE)
    ax.set_xticks(x,t.metric); ax.set_ylim(0,1.12); ax.legend(facecolor=NAVY,labelcolor=WHITE); save(fig,'slide_09_validation',t)

def net(s):
    """Net sentiment among recognized labels; neutral and mixed stay in denominator."""
    s=s[s.isin(SENTIMENTS)]
    return 100*((s=='positive').sum()-(s=='negative').sum())/len(s) if len(s) else np.nan

def aspects(d):
    rows=[]
    for f in FEATURES:
        s=d.loc[flag(d[f+'_llm']),f+'_sentiment_llm']; known=s[s.isin(SENTIMENTS)]
        rows.append({'feature':f,'mentions':len(s),'classified':len(known),'negative_pct':100*(known=='negative').mean() if len(known) else np.nan,'net':net(s)})
    return pd.DataFrame(rows).set_index('feature')

def drivers(d):
    all_=aspects(d).sort_values('mentions',ascending=False).head(8); ae=aspects(d[d.airline.eq(AE)]).loc[all_.index]
    fig,axes=canvas('Passenger-experience drivers | ranked by mentions',2,(16,7))
    for ax,t,title in zip(axes,[all_,ae],[f'All airlines | {len(d):,}',f'AEGEAN | {d.airline.eq(AE).sum():,}']):
        ax.set_title(title); horizontal(ax,[label(f) for f in t.index],t.mentions,fmt=lambda v:f'{v:,.0f}')
        for i,v in enumerate(t.negative_pct): ax.text(.99,i,f'{v:.0f}% neg',transform=ax.get_yaxis_transform(),ha='right',va='center',color=RED)
        ax.set_xlim(0,max(t.mentions.max(),1)*1.55)
    save(fig,'slide_10_drivers',pd.concat([all_.assign(group='All'),ae.assign(group='AEGEAN')]).reset_index())

def overall(d):
    rows=[]
    for a,g in d.groupby('airline'):
        s=g.gemini_overall_sentiment; p=s.value_counts(normalize=True)*100
        rows.append({'airline':a,'n':len(g),'positive':p.get('positive',0),'negative':p.get('negative',0),'neutral_mixed':p.get('neutral',0)+p.get('mixed',0)})
    t=pd.DataFrame(rows).set_index('airline'); fig,axes=canvas('2024–2026 Gemini sentiment | two benchmark groups',2,(16,6))
    for ax,group,title in zip(axes,[EUROPE,REGIONAL],['European benchmark','Global regional benchmark']):
        v=t.reindex(group).dropna(subset=['n']); left=np.zeros(len(v)); ax.set_title(title)
        for col,color in [('positive',GREEN),('negative',RED),('neutral_mixed',GRAY)]:
            b=ax.barh(v.index,v[col],left=left,color=color,label=col.replace('_',' / ')); ax.bar_label(b,labels=[f'{x:.1f}%' if x>=8 else '' for x in v[col]],label_type='center',color=WHITE); left+=v[col].to_numpy()
        for i,n in enumerate(v.n): ax.text(102,i,f'{n:,.0f}',va='center')
        ax.set_xlim(0,120); ax.set_xticks([]); ax.invert_yaxis(); ax.legend(loc='lower center',bbox_to_anchor=(.5,-.18),ncol=3,facecolor=NAVY,labelcolor=WHITE,fontsize=9)
    save(fig,'slide_11_overall_benchmark',t.reset_index())

def peers(d,group,slide):
    tables={a:aspects(d[d.airline.eq(a)]) for a in group}; rows=[]
    for f in FEATURES:
        a=tables[AE].loc[f]; candidates=[(p,tables[p].loc[f]) for p in group[1:] if pd.notna(tables[p].loc[f,'net'])]
        if not candidates or pd.isna(a.net): continue
        p,b=max(candidates,key=lambda x:x[1].net)
        rows.append({'feature':f,'peer':p,'aegean_net':a.net,'peer_net':b.net,'aegean_n':a.mentions,'peer_n':b.mentions,'gap':a.net-b.net})
    t=pd.DataFrame(rows,columns=['feature','peer','aegean_net','peer_net','aegean_n','peer_n','gap']).sort_values('gap',ascending=False)
    v=pd.concat([t.head(2),t.tail(2).sort_values('gap')]).drop_duplicates('feature'); fig,(ax,)=canvas(f'{"European" if slide==12 else "Regional"} benchmark | AEGEAN vs best peer')
    ax.axis('off'); cells=[[label(r.feature),f'{r.aegean_net:+.1f} pp\n(n={r.aegean_n:,})',f'{r.peer}\n{r.peer_net:+.1f} pp (n={r.peer_n:,})',f'{r.gap:+.1f} pp'] for r in v.itertuples()]
    table(ax,['Feature','AEGEAN','Best competitor','Gap'],cells,[.25,.2,.35,.2]); ax.set_title(f'{(t.gap>0).sum()} of {len(t)} comparable features beat the best peer',color=GOLD)
    save(fig,f'slide_{slide}_best_peer',t.assign(displayed=t.feature.isin(v.feature)))

def table(ax, headers, cells, widths):
    t=ax.table(cellText=cells or [['—']*len(headers)],colLabels=headers,colWidths=widths,cellLoc='center',bbox=[0,.04,1,.88]); t.auto_set_font_size(False); t.set_fontsize(12)
    for (i,j),c in t.get_celld().items():
        c.set_facecolor('#193858' if i%2 else '#142E4D'); c.set_edgecolor(NAVY); c.get_text().set_color(GOLD if i==0 else WHITE)
        if i and j>0:
            text=c.get_text().get_text().strip()
            if text.startswith('+'): c.get_text().set_color(GREEN)
            elif text.startswith('-') and j==len(headers)-1: c.get_text().set_color(RED)
    return t

def evolution(d):
    rows=[]
    for a in AIRLINES:
        g=d[d.airline.eq(a)]; prior=net(g.loc[g.period.eq('2024–2025'),'gemini_overall_sentiment']); recent=net(g.loc[g.period.eq('2026'),'gemini_overall_sentiment']); rows.append({'airline':a,'prior':prior,'recent':recent,'delta':recent-prior})
    t=pd.DataFrame(rows).set_index('airline'); fig,axes=canvas('Sentiment evolution | 2024–2025 vs 2026 YTD',2,(16,6))
    for ax,group,title in zip(axes,[EUROPE,REGIONAL],['European benchmark','Global regional benchmark']):
        ax.axis('off'); ax.set_title(title); table(ax,['Airline','2024–25','2026','Δ pp'],[[a,*[f'{x:+.1f}' if pd.notna(x) else '—' for x in t.loc[a]]] for a in group],[.4,.2,.2,.2])
    save(fig,'slide_14_evolution',t.reset_index())

def changes(d):
    rows=[]; cells=[]
    for a in AIRLINES:
        g=d[d.airline.eq(a)]; p=aspects(g[g.period.eq('2024–2025')]); r=aspects(g[g.period.eq('2026')]); t=pd.DataFrame({'prior_n':p.mentions,'recent_n':r.mentions,'prior_net':p.net,'recent_net':r.net}); t['delta']=t.recent_net-t.prior_net; t['eligible']=t.prior_n.ge(20)&t.recent_n.ge(20)&t.delta.notna(); v=t[t.eligible]; up=v[v.delta.gt(0)].nlargest(2,'delta'); down=v[v.delta.lt(0)].nsmallest(2,'delta'); t['displayed']=t.index.isin(up.index.union(down.index)); rows.append(t.assign(airline=a).reset_index()); cells.append([a,'\n'.join(f'{label(f)}  {x.delta:+.1f} pp' for f,x in up.iterrows()) or '—','\n'.join(f'{label(f)}  {x.delta:+.1f} pp' for f,x in down.iterrows()) or '—'])
    fig,(ax,)=canvas('Which passenger-experience themes changed most?',size=(15,8)); ax.axis('off'); tab=table(ax,['Airline','Top 2 improvements ↑','Top 2 deteriorations ↓'],cells,[.25,.375,.375]);
    for (i,j),c in tab.get_celld().items():
        if i and j: c.get_text().set_color(GREEN if j==1 else RED)
    ax.set_title('At least 20 feature mentions per airline in each period',color=GOLD); save(fig,'slide_15_feature_changes',pd.concat(rows,ignore_index=True))

def main():
    global OUTPUT
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--hf',type=Path,default=HF); parser.add_argument('--gemini',type=Path,default=GEMINI); parser.add_argument('--output',type=Path,default=OUTPUT); parser.add_argument('--exclude-long-haul-first-class',action='store_true'); args=parser.parse_args(); OUTPUT=args.output; OUTPUT.mkdir(parents=True,exist_ok=True)
    d=prepare(args.hf,args.gemini,args.exclude_long_haul_first_class)
    eda(d); validation(); drivers(d); overall(d); peers(d,EUROPE,12); peers(d,REGIONAL,13); evolution(d); changes(d)
    print(f'All eight visual sets saved in {OUTPUT}')

if __name__=='__main__': main()
