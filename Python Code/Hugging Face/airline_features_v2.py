"""Conservative, auditable airline-review feature engineering. Python 3.10+."""
from __future__ import annotations
import re, math, json, hashlib
from pathlib import Path
from functools import lru_cache
from collections import defaultdict
import numpy as np
import pandas as pd

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if all((p / n).is_dir() for n in ("Python Code", "Data", "Streamlit"))), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Place this module inside the AEGEAN project tree.")
ROOT = PROJECT_ROOT / "Data" / "Output" / "Hugging Face"
CLEANING_DIR = PROJECT_ROOT / "Data" / "Output" / "Cleaning data"
FEATURES = ['long_haul','first_class','cancellation','delay','boarding','baggage','economy_class','business_class','international','domestic','customer_service','cabin_crew','ground_staff','complaint_resolution','seat_comfort','food_beverage','cabin_cleanliness','entertainment_wifi','booking_experience','check_in','airport_experience','value_for_money','pricing_fees','flight_disruption','overall_experience']
REGIONAL = {'Aegean Airlines','Bangkok Air','Porter Airlines','Virgin Australia'}
# European benchmark: AEGEAN plus the three European competitors.
EUROPEAN = {'Aegean Airlines','Air France','Lufthansa','Turkish Airlines'}
SEED = 42
# Evidence-strength scores are deterministic heuristics, NOT probabilities.
STRENGTH = {'verified':1.0,'explicit':0.90,'route':0.80,'context':0.75,'term':0.65,'unknown':0.0}

# Each rule uses explicit phrase contexts; ambiguity is intentionally handled conservatively.
PATTERNS = {
 'first_class': [r'\b(?:i|we) (?:flew|travelled|traveled|sat|were seated) (?:in )?first[ -]class\b',r'\bmy first[ -]class (?:seat|cabin|ticket)\b',r'\b(?:upgraded|upgrade) to first[ -]class\b',r'\b(?:booked|purchased|bought) (?:a |the )?first[ -]class (?:ticket|seat|fare)\b'],
 'business_class':[r'\b(?:i|we) (?:flew|travelled|traveled|sat|were seated) (?:in )?business[ -]class\b',r'\bmy business[ -]class (?:seat|cabin|ticket)\b',r'\b(?:upgraded|upgrade) to business[ -]class\b',r'\b(?:booked|purchased|bought) (?:a |the )?business[ -]class (?:ticket|seat|fare)\b'],
 'economy_class':[r'\b(?:i|we) (?:flew|travelled|traveled|sat|were seated) (?:in )?(?:economy(?:[ -]class)?|coach)\b',r'\bmy economy(?:[ -]class)? (?:seat|cabin|ticket)\b',r'\b(?:booked|purchased|bought) (?:an? |the )?economy(?:[ -]class)? (?:ticket|seat|fare)\b'],
 'cancellation':[r'\b(?:my|our|the) flight (?:was|got|has been|had been) cancel(?:led|ed)\b',r'\b(?:airline|carrier) cancel(?:led|ed) (?:my|our|the) flight\b',r'\b(?:received|got) (?:a |the )?flight cancellation (?:notice|notification|email)\b',r'\b(?:my|our|the) flight was called off\b'],
 'delay':[r'\b(?:my|our|the) flight (?:was|got|had been|has been) delay(?:ed)?\b',r'\b(?:our|the|my) (?:departure|arrival|flight) (?:was|is) \w*(?:\s\w+){0,2} (?:hours?|minutes?) late\b',r'\b(?:we|i) arrived \w*(?:\s\w+){0,2} (?:hours?|minutes?) (?:late|behind schedule)\b',r'\bmissed (?:my|our|the) connection due to (?:the |a )?delay\b',r'\b(?:flight|departure|arrival) delay(?:s|ed)?\b'],
 'boarding':[r'\bboarding (?:process|queue|line|procedure|was|is|started|began|took|gate|experience)\b',r'\bpriority boarding\b',r'\b(?:we|i) boarded (?:the |our |a )?(?:plane|aircraft|flight)\b',r'\b(?:gate|aircraft) boarding\b'],
 'baggage':[r'\b(?:my|our|the) (?:luggage|baggage|suitcase|bag|carry[ -]?on)\b',r'\b(?:checked|lost|delayed|damaged|missing) (?:baggage|luggage|bags?|suitcases?)\b',r'\b(?:baggage|luggage|carry[ -]?on|hand luggage) (?:allowance|fees?|charges?|claim|policy|handling)\b',r'\boverhead bins?\b',r'\b(?:paid|charged) (?:extra|more) for (?:a |my |our )?(?:bag|baggage|luggage)\b'],
 'international':[r'\b(?:an? |our |my |the )?international (?:flight|route)\b',r'\bcross[ -]border flight\b'],
 'domestic':[r'\b(?:a |our |my |the )?domestic (?:flight|route)\b',r'\bflight within (?:the|my|our) country\b']
}
# Topic evidence must concern a passenger experience, not a generic passing reference.
EXTRA_PATTERNS = {
 'customer_service':[r'\b(?:customer (?:service|support|care)|call cent(?:er|re)|support (?:team|agent|staff)|helpline|service desk)\b'],
 'cabin_crew':[r'\b(?:cabin crew|flight attendants?|steward(?:ess|s)?|onboard crew|in[- ]flight crew|purser)\b'],
 'ground_staff':[r'\b(?:ground staff|ground crew|gate agents?|check[- ]in (?:agents?|staff)|airport (?:agents?|staff|representatives?))\b'],
 'complaint_resolution':[r'\b(?:refund|reimbursement|compensation|complaint|claim|dispute|chargeback)\b.{0,65}\b(?:resolved|denied|refused|ignored|processed|pending|rejected|handled|response|received|paid|waiting|unanswered)\b',r'\b(?:resolved|denied|refused|ignored|processed|pending|rejected|handled|received|waiting for)\b.{0,65}\b(?:refund|compensation|complaint|claim|reimbursement)\b'],
 'seat_comfort':[r'\b(?:seat|seats|legroom|recline|seat pitch|knee room)\b.{0,55}\b(?:comfortable|uncomfortable|cramped|spacious|tight|narrow|hard|broken|roomy|recline|space|width|painful)\b',r'\b(?:comfortable|uncomfortable|cramped|spacious|tight|narrow|hard|broken|roomy)\b.{0,35}\bseats?\b',r'\b(?:lack of legroom|plenty of legroom|no legroom)\b'],
 'food_beverage':[r'\b(?:meal|meals|food|snack|snacks|breakfast|lunch|dinner|catering|drinks?|beverages?|coffee|tea|water)\b.{0,55}\b(?:served|service|quality|taste|delicious|terrible|good|bad|free|included|paid|offered|available|cold|hot|stale|portion)\b',r'\b(?:served|offered|bought|provided|ate|drank)\b.{0,35}\b(?:food|meal|snack|drink|beverage|breakfast|lunch|dinner)\b'],
 'cabin_cleanliness':[r'\b(?:cabin|aircraft|plane|seat|toilet|lavatory|tray table|interior)\b.{0,50}\b(?:clean|dirty|filthy|spotless|hygienic|unclean|stained|smelly|messy)\b',r'\b(?:clean|dirty|filthy|spotless|unclean|stained|smelly)\b.{0,35}\b(?:cabin|aircraft|plane|seat|toilet|lavatory)\b'],
 'entertainment_wifi':[r'\b(?:in[- ]flight entertainment|seatback screen|onboard wi[- ]?fi|in[- ]flight wi[- ]?fi|onboard internet|onboard connectivity|charging ports?|usb ports?|power outlets?)\b',r'\b(?:wi[- ]?fi|internet|entertainment|movies?|screens?)\b.{0,45}\b(?:on board|onboard|flight|aircraft|seat|cabin|worked|broken|available|unavailable)\b'],
 'booking_experience':[r'\b(?:booking|reservation|booked|ticket purchase|payment|website|mobile app|airline app|online portal)\b.{0,65}\b(?:failed|error|problem|issue|easy|difficult|smooth|confusing|changed|change|charged|paid|crashed|worked|process)\b',r'\b(?:trouble|problem|issue|difficulty|easy|difficult)\b.{0,40}\b(?:booking|reservation|payment|website|app)\b'],
 'check_in':[r'\b(?:check[- ]in|checked in|checkin)\b.{0,55}\b(?:counter|desk|queue|process|online|kiosk|staff|agent|slow|fast|smooth|failed|problem|issue|time|open|closed)\b',r'\b(?:online|airport|self[- ]service) check[- ]in\b'],
 'airport_experience':[r'\b(?:airport lounge|business lounge|terminal transfer|airport assistance|airport facilities|lounge access|airport experience)\b',r'\b(?:lounge|terminal)\b.{0,50}\b(?:crowded|comfortable|dirty|clean|excellent|poor|access|facilities|staff|service)\b'],
 'value_for_money':[r'\b(?:value for money|worth (?:the|its|every) (?:price|money|penny|cost)|not worth (?:the|its) (?:price|money|cost)|good value|poor value|overpriced for|worth what (?:i|we) paid)\b'],
 'pricing_fees':[r'\b(?:ticket price|airfare|fare price|hidden fees?|extra charges?|seat selection fees?|baggage fees?|change fees?|additional fees?|unexpected charges?|ancillary fees?|expensive tickets?|cheap tickets?)\b',r'\b(?:charged|paid|cost)\b.{0,40}\b(?:extra|fee|fees|seat selection|baggage|luggage)\b'],
 'flight_disruption':[r'\b(?:missed (?:my|our|the|a) connection|missed connecting flight|diverted flight|flight diversion|rerouted|rebooked|rebooking|stranded|disrupted itinerary|overnight accommodation due to)\b'],
 'overall_experience':[r'\b(?:overall (?:experience|flight|journey|trip)|whole experience|entire experience|in general (?:the|this) (?:airline|flight)|would (?:fly|travel|book) (?:with )?(?:them|this airline) again|never (?:fly|travel|book) (?:with )?(?:them|this airline) again|highly recommend (?:this|the) airline)\b']
}
PATTERNS.update(EXTRA_PATTERNS)
COMPILED = {k:[re.compile(x,re.I) for x in v] for k,v in PATTERNS.items()}
NEGATION = re.compile(r'\b(?:not|never|no|without|neither|wasn.t|weren.t|didn.t|isn.t|couldn.t)\b',re.I)
HYPOTHETICAL = re.compile(r'\b(?:might|could|would|if|worried|feared|expected|hypothetical|potentially)\b',re.I)
DURATION = re.compile(r'\b(?P<h>\d{1,2}(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b(?:\s*(?:and)?\s*(?P<m>\d{1,2})\s*(?:minutes?|mins?|m)\b)?',re.I)
DURATION_M = re.compile(r'\b(?P<h>\d{1,2})h\s*(?P<m>\d{1,2})m\b',re.I)
LONG_TERMS = re.compile(r'\b(?:long[ -]haul|intercontinental|transatlantic|transpacific) (?:flight|route|journey)\b',re.I)
FLIGHT_CONTEXT = re.compile(r'\b(?:flight|flying|flew|airborne|in the air|inflight|in-flight|plane|aircraft|segment|leg)\b',re.I)
NON_FLIGHT_DURATION = re.compile(r'\b(?:delay(?:ed)?|wait(?:ed|ing)?|layover|stopover|connection|transfer|airport|queue|response|customer service|total journey|including connections|travel time|journey including)\b',re.I)
ROUTE_CONNECTOR = re.compile(r'\b(?:from|between)\s+([a-z][a-z .-]{1,35}?)\s+(?:to|and)\s+([a-z][a-z .-]{1,35}?)(?=[,.;!?]|\s+(?:and then|then|on|with|where|for|in|was|is|by|after|before)|$)',re.I)
IATA_ROUTE = re.compile(r'\b([A-Z]{3})\s*(?:-|–|—|→|/| to )\s*([A-Z]{3})\b')

# Optional reliable airport reference; install: pip install airportsdata
try:
 import airportsdata
 AIRPORTS = airportsdata.load('IATA')
except ImportError:
 AIRPORTS = {}

# City aliases intentionally restricted to unambiguous names; airport reference supplements these.
CITY_CODES = {'athens':'ATH','thessaloniki':'SKG','london':'LHR','paris':'CDG','istanbul':'IST','bangkok':'BKK','rhodes':'RHO','new york':'JFK','frankfurt':'FRA','munich':'MUC','rome':'FCO','sydney':'SYD','melbourne':'MEL','toronto':'YYZ','montreal':'YUL','vancouver':'YVR','singapore':'SIN','los angeles':'LAX','chicago':'ORD','dubai':'DXB'}
# Fallback for tests when airportsdata is absent; operational use should install airportsdata.
FALLBACK = {'ATH':(37.9364,23.9445,'GR'),'SKG':(40.5197,22.9709,'GR'),'LHR':(51.4700,-0.4543,'GB'),'CDG':(49.0097,2.5479,'FR'),'IST':(41.2753,28.7519,'TR'),'BKK':(13.6900,100.7501,'TH'),'RHO':(36.4054,28.0862,'GR'),'JFK':(40.6413,-73.7781,'US'),'FRA':(50.0379,8.5622,'DE'),'MUC':(48.3538,11.7861,'DE'),'FCO':(41.8003,12.2389,'IT'),'SYD':(-33.9399,151.1753,'AU'),'MEL':(-37.6690,144.8410,'AU'),'YYZ':(43.6777,-79.6248,'CA'),'YUL':(45.4706,-73.7408,'CA'),'YVR':(49.1967,-123.1815,'CA'),'SIN':(1.3644,103.9915,'SG'),'LAX':(33.9416,-118.4085,'US'),'ORD':(41.9742,-87.9073,'US'),'DXB':(25.2532,55.3657,'AE')}

def normalize_text(s):
 """Return or process normalize text."""
 return re.sub(r'\s+',' ',str(s)).strip() if pd.notna(s) else ''

def airport(code):
 """Return or process airport."""
 code = str(code).upper()
 d = AIRPORTS.get(code)
 if d:
  return (float(d['lat']),float(d['lon']),str(d['country']))
 return FALLBACK.get(code)

def resolve_place(value):
 """Return or process resolve place."""
 s = normalize_text(value).strip(' ,.;').lower()
 if not s:return None
 if len(s)==3 and (s.upper() in AIRPORTS or s.upper() in FALLBACK):return s.upper()
 if s in CITY_CODES:return CITY_CODES[s]
 # AirportsData's city names can be non-unique; avoid guessing ambiguous matches.
 return None

def haversine(a,b):
 """Return or process haversine."""
 lat1,lon1 = map(math.radians,a[:2]);lat2,lon2 = map(math.radians,b[:2])
 x=math.sin((lat2-lat1)/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
 return 6371*2*math.asin(min(1,math.sqrt(x)))

@lru_cache(maxsize=30000)
def route_info(a,b):
 """Return or process route info."""
 aa,bb=airport(a),airport(b)
 if not aa or not bb:return None
 km=haversine(aa,bb)
 # Approximate block time: 45 minutes ground/terminal overhead + 780 km/h cruise.
 # ±45-minute band is intentionally conservative; not a scheduled duration.
 estimate=45+60*km/780
 return {'km':round(km,1),'minutes':round(estimate,1),'low':round(estimate-45,1),'high':round(estimate+45,1),'international':aa[2]!=bb[2]}

def segments(row,text):
 """Return or process segments."""
 result=[]
 def add(a,b,method):
  """Return or process add."""
  a,b=resolve_place(a),resolve_place(b)
  if a and b and a!=b and (a,b) not in [(x[0],x[1]) for x in result]:result.append((a,b,method))
 if normalize_text(row.get('origin')) and normalize_text(row.get('destination')):
  add(row.get('origin'),row.get('destination'),'structured')
 route=normalize_text(row.get('route'))
 for s in [route,text]:
  for m in IATA_ROUTE.finditer(s):add(m.group(1),m.group(2),'route_text')
  for m in ROUTE_CONNECTOR.finditer(s):add(m.group(1),m.group(2),'route_text')
 # Identify explicit city-to-city pairs even in connecting itineraries.
 for s in [route,text]:
  for city_a,code_a in CITY_CODES.items():
   for city_b,code_b in CITY_CODES.items():
    if city_a==city_b:continue
    p=re.compile(r'\b'+re.escape(city_a)+r'\s+(?:to|–|—|-)\s+'+re.escape(city_b)+r'\b',re.I)
    if p.search(s):add(code_a,code_b,'route_text')
 return result

def record(feature,state='unknown',method='none',evidence='',source='',strength=0,uncertain=True,conflict=False,minutes=np.nan,segments_count=0):
 """Return or process record."""
 return {'feature':feature,'binary_prediction':int(state=='positive'),'internal_classification_state':state,'classification_method':method,'matched_evidence':str(evidence)[:300],'evidence_source':source,'confidence_score':strength,'uncertainty_flag':bool(uncertain),'conflict_flag':bool(conflict),'estimated_flight_duration_minutes':minutes,'identified_segments':segments_count}

def sentence_for(text,start,end):
 """Return or process sentence for."""
 left=max(text.rfind('.',0,start),text.rfind('!',0,start),text.rfind('?',0,start),text.rfind(';',0,start))+1
 ends=[i for i in (text.find(c,end) for c in '.!?;') if i>=0]
 right=min(ends) if ends else len(text)
 return text[left:right].strip()

def contextual(feature,text):
 """Return or process contextual."""
 candidates=[];excluded=[]
 for p in COMPILED[feature]:
  for m in p.finditer(text):
   clause=sentence_for(text,m.start(),m.end())
   pre=text[max(0,m.start()-45):m.start()]
   # Negation close to phrase, plus explicit cancellation and delay negation idioms.
   neg=bool(NEGATION.search(pre[-22:])) or bool(re.search(r'\b(?:no|without any|not|never)\s+(?:flight )?(?:delay|cancellation|boarding|baggage)\b',clause,re.I))
   if feature=='delay' and re.search(r'\b(?:no delays?|without any delay|on time|not delayed)\b',clause,re.I):neg=True
   if feature=='cancellation' and re.search(r'\b(?:not cancel(?:led|ed)|never cancel(?:led|ed)|might be cancel(?:led|ed))\b',clause,re.I):neg=True
   hypo=bool(HYPOTHETICAL.search(pre[-35:]))
   if neg or hypo:excluded.append(m.group());continue
   candidates.append(m.group())
 if candidates:return record(feature,'positive','context_regex',candidates[0],'review_en',STRENGTH['context'],False)
 if excluded:return record(feature,'negative','negated_or_hypothetical',excluded[0],'review_en',STRENGTH['explicit'],False)
 return record(feature)

def duration_evidence(text):
 """Return or process duration evidence."""
 positive=[];negative=[];ambiguous=[]
 for pat in (DURATION_M,DURATION):
  for m in pat.finditer(text):
   minutes=round(float(m.group('h'))*60+float(m.group('m') or 0))
   clause=sentence_for(text,m.start(),m.end())
   near=text[max(0,m.start()-65):min(len(text),m.end()+65)]
   if NON_FLIGHT_DURATION.search(clause):
    ambiguous.append(m.group());continue
   if not FLIGHT_CONTEXT.search(clause):
    ambiguous.append(m.group());continue
   if minutes>360:positive.append((minutes,m.group()))
   else:negative.append((minutes,m.group()))
 return positive,negative,ambiguous

def classify_long(text,segs):
 """Return or process classify long."""
 routes=[]
 for a,b,src in segs:
  info=route_info(a,b)
  if info:routes.append((a,b,src,info))
 positives,negatives,ambiguous=duration_evidence(text)
 # Explicit segment flight-time statements take priority over approximate route times.
 if positives:
  v=positives[0];conflict=bool(negatives) or any(x[3]['high']<=360 for x in routes)
  return record('long_haul','positive','explicit_flight_duration',v[1],'review_en',STRENGTH['explicit'],conflict,conflict,v[0],len(segs))
 high=[x for x in routes if x[3]['low']>360]
 low=[x for x in routes if x[3]['high']<=360]
 if high:
  x=high[0];conflict=bool(negatives)
  return record('long_haul','positive','route_estimate',f'{x[0]}-{x[1]} ~{x[3]["minutes"]}min (±45min)',x[2],STRENGTH['route'],conflict,conflict,x[3]['minutes'],len(segs))
 if LONG_TERMS.search(text):
  m=LONG_TERMS.search(text);conflict=bool(negatives or low)
  if conflict:return record('long_haul','ambiguous','conflicting_terminology',m.group(),'review_en',0,True,True,np.nan,len(segs))
  return record('long_haul','positive','explicit_long_haul_term',m.group(),'review_en',STRENGTH['term'],True,False,np.nan,len(segs))
 if negatives and not ambiguous and not routes:return record('long_haul','negative','explicit_short_duration',negatives[0][1],'review_en',STRENGTH['explicit'],False,False,negatives[0][0],len(segs))
 if low and len(low)==len(routes) and not ambiguous:return record('long_haul','negative','route_estimate',f'{low[0][0]}-{low[0][1]}',low[0][2],STRENGTH['route'],True,False,low[0][3]['minutes'],len(segs))
 return record('long_haul','unknown','insufficient_duration_evidence','; '.join(ambiguous[:2]),'review_en' if ambiguous else '',0,True,False,np.nan,len(segs))

def classify_geography(text,segs):
 """Return or process classify geography."""
 infos=[(a,b,src,route_info(a,b)) for a,b,src in segs]
 infos=[x for x in infos if x[3]]
 out={}
 for feature,desired in [('international',True),('domestic',False)]:
  hits=[x for x in infos if x[3]['international']==desired]
  if hits:
   x=hits[0];out[feature]=record(feature,'positive','airport_countries',f'{x[0]}-{x[1]}',x[2],STRENGTH['verified'],False,False,np.nan,len(segs))
  elif infos:
   out[feature]=record(feature,'negative','airport_countries',f'{infos[0][0]}-{infos[0][1]}',infos[0][2],STRENGTH['verified'],False,False,np.nan,len(segs))
  else:out[feature]=contextual(feature,text)
 if out['international']['binary_prediction'] and out['domestic']['binary_prediction'] and len(infos)<2:
  for f in ('international','domestic'):out[f]['conflict_flag']=True
 return out

def classify_one(row):
 """Return or process classify one."""
 text=normalize_text(row.get('review_en'))
 segs=segments(row,text)
 d={f:contextual(f,text) for f in ('first_class','business_class','economy_class','cancellation','delay','boarding','baggage',*EXTRA_PATTERNS)}
 d['long_haul']=classify_long(text,segs)
 d.update(classify_geography(text,segs))
 cabin=[f for f in ('first_class','business_class','economy_class') if d[f]['binary_prediction']]
 if len(cabin)>1:
  for f in cabin:d[f]['conflict_flag']=True
 return d

def load_data():
 """Return or process load data."""
 if 'df_final' in globals() and isinstance(globals()['df_final'],pd.DataFrame):return globals()['df_final'].copy(deep=True)
 return pd.read_csv(CLEANING_DIR/'df_final.csv',encoding='utf-8-sig',low_memory=False)

def run_pipeline(df,output_dir=ROOT,save=True):
 """Return or process run pipeline."""
 required=['record_key','airline','source','review_en']
 missing=[x for x in required if x not in df.columns]
 if missing:raise ValueError(f'Missing required columns: {missing}')
 if df['record_key'].isna().any() or not df['record_key'].is_unique:raise ValueError('record_key must be unique and non-null')
 original=df.copy(deep=True)
 result=df.copy(deep=True)
 diag=[]
 from tqdm.auto import tqdm
 # Deduplicate identical classification inputs without conflating different structured routes.
 cache={}
 for _,r in tqdm(df.iterrows(),total=len(df),desc='Classifying reviews'):
  sig=(normalize_text(r.get('review_en')),normalize_text(r.get('origin')),normalize_text(r.get('destination')),normalize_text(r.get('route')),normalize_text(r.get('flight_number')))
  if sig not in cache:cache[sig]=classify_one(r)
  predictions=cache[sig]
  diag.extend([{'record_key':r['record_key'],**predictions[f]} for f in FEATURES])
 diagnostics=pd.DataFrame(diag)
 wide=diagnostics.pivot(index='record_key',columns='feature',values='binary_prediction')
 for f in FEATURES:result[f]=result['record_key'].map(wide[f]).astype('int8')
 result['regional'] = result['airline'].astype(str).str.strip().isin(REGIONAL).astype('int8')
 result['european'] = result['airline'].astype(str).str.strip().isin(EUROPEAN).astype('int8')
 filtered = result.loc[(result['first_class']==0) & (result['long_haul']==0)].copy()
 assert len(result)==len(original) and result['record_key'].equals(original['record_key'])
 pd.testing.assert_frame_equal(result[original.columns],original,check_dtype=True)
 assert all(result[f].isin([0,1]).all() for f in FEATURES)
 assert len(diagnostics)==len(result)*len(FEATURES)
 nonempty=result['review_en'].fillna('').astype(str).str.strip().ne('')
 summaries=[]
 for f in FEATURES:
  x=diagnostics[diagnostics.feature==f]
  n=int(result[f].sum())
  summaries.append({'feature':f,'positive_count':n,'percentage_all':round(100*n/len(result),2),'percentage_nonempty':round(100*result.loc[nonempty,f].sum()/max(1,nonempty.sum()),2),'uncertain_count':int(x.uncertainty_flag.sum()),'manual_review_count':int((x.uncertainty_flag|x.conflict_flag).sum())})
 summary=pd.DataFrame(summaries)
 def grouped(col):
  """Return or process grouped."""
  counts=result.groupby(col,dropna=False)[FEATURES].agg(['sum','mean'])
  counts.columns=[f'{f}_{"count" if agg=="sum" else "percentage"}' for f,agg in counts.columns]
  for f in FEATURES:counts[f+'_percentage']=(counts[f+'_percentage']*100).round(2)
  return counts.reset_index()
 by_airline=grouped('airline');by_source=grouped('source')
 conflicts=diagnostics.loc[diagnostics.conflict_flag].copy()
 # Sample 20 positives and 20 zeros for each feature with a fixed seed.
 validation=[]
 indexed=result.set_index('record_key',drop=False)
 for f in FEATURES:
  d=diagnostics[diagnostics.feature==f]
  for label in [1,0]:
   subset=d[d.binary_prediction==label].sample(n=min(20,(d.binary_prediction==label).sum()),random_state=SEED)
   for r in subset.to_dict('records'):
    base=indexed.loc[r['record_key']]
    validation.append({**r,'airline':base['airline'],'source':base['source'],'review_en':base['review_en'],'manual_label':''})
 df_validation=pd.DataFrame(validation)
 if save:
  p=Path(output_dir);p.mkdir(parents=True,exist_ok=True)
  files={'df_final_features.csv':result,'df_final_features_no_first_class_no_long_haul.csv':filtered,'classification_diagnostics.csv':diagnostics,'feature_summary.csv':summary,'feature_summary_by_airline.csv':by_airline,'feature_summary_by_source.csv':by_source,'classification_validation_sample.csv':df_validation,'classification_conflicts.csv':conflicts}
  for name,data in files.items():data.to_csv(p/name,index=False,encoding='utf-8-sig')
 return result,filtered,diagnostics,summary,by_airline,by_source,df_validation,conflicts

def run_tests():
 """Return or process run tests."""
 tests=[('Our flight was delayed by three hours.','delay',1),('The flight departed without any delay.','delay',0),('I traveled in business class.','business_class',1),('I was traveling for business.','business_class',0),('Business class felt like economy.','economy_class',0),('The boarding process was excellent.','boarding',1),('My flight lasted 8 hours.','long_haul',1),('The layover lasted 8 hours.','long_haul',0),('I flew first class.','first_class',1),('The service was first class.','first_class',0),('My luggage was lost.','baggage',1),('My flight was not cancelled.','cancellation',0),('I flew from Athens to London.','international',1),('I flew from Athens to Thessaloniki.','domestic',1),('Our flight was cancelled.','cancellation',1),('I flew economy.','economy_class',1),('Our flight was exactly 6 hours.','long_haul',0),('The flight was delayed by 8 hours.','long_haul',0),('Athens to Istanbul and then Istanbul to Bangkok.','long_haul',1)]
 errors=[]
 for t,f,expected in tests:
  row={'review_en':t,'origin':'','destination':'','route':''}
  actual=classify_one(row)[f]['binary_prediction']
  if actual!=expected:errors.append((t,f,expected,actual))
 return pd.DataFrame(errors,columns=['text','feature','expected','actual'])

def evaluate_manual_labels(df_validation):
 """Requires human-completed manual_label 0/1. Never invent labels or metrics."""
 from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
 d=df_validation.copy();d['manual_label']=pd.to_numeric(d['manual_label'],errors='coerce')
 d=d[d.manual_label.isin([0,1])]
 rows=[]
 for f,g in d.groupby('feature'):
  p,r,f1,_=precision_recall_fscore_support(g.manual_label.astype(int),g.binary_prediction.astype(int),average='binary',zero_division=0)
  rows.append({'feature':f,'labeled_n':len(g),'precision':p,'recall':r,'f1':f1,'confusion_matrix':confusion_matrix(g.manual_label,g.binary_prediction,labels=[0,1]).tolist()})
 return pd.DataFrame(rows)

if __name__=='__main__':
 errors=run_tests()
 if not errors.empty:print('WARNING: Tests needing review:\n',errors.to_string(index=False))
 else:print('All automated tests passed')
 data=load_data()
 outputs=run_pipeline(data)
 print(outputs[3].to_string(index=False))
