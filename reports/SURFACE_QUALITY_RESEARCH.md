# Surface Quality And Distress Classification Research

This report uses only the adversarially verified claims in `/tmp/tarmac_research.json` as its factual backbone. Notes labeled **Engineering interpretation** are project guidance for Tarmac and should not be read as additional standard text.

## Source Base

- FHWA LTPP Distress Identification Manual, 2014: https://www.fhwa.dot.gov/publications/research/infrastructure/pavements/ltpp/13092/13092.pdf
- FHWA LTPP Distress Identification Manual, 2003 report copy: https://www.fhwa.dot.gov/publications/research/infrastructure/pavements/ltpp/reports/03031/03031.pdf
- ASTM D6433-20 store page: https://store.astm.org/d6433-20.html
- TxDOT visual pavement condition survey manual: https://www.txdot.gov/manuals/mnt/pdm/pavement_evaluation/visual_p_cond_surveys-i1004115.html
- FHWA Specifications for the National Bridge Inventory, March 2022: https://www.fhwa.dot.gov/bridge/snbi/snbi_march_2022_publication.pdf
- AASHTO bridge element guide manual copy: https://apmgs.ro/files/documente/AASHTO-bridge_element_guide_manual__05092010.pdf

## A. Methodology And Quality Attributes

The FHWA LTPP Distress Identification Manual is the verified U.S. reference for common pavement distress taxonomy, severity levels, and measurement methods. It is organized into asphalt concrete-surfaced pavement, jointed portland cement concrete, and continuously reinforced portland cement concrete, with distress groups such as cracking, surface deformation, surface defects, and miscellaneous distresses. Each distress has a measurement unit, and only some distress types have defined severity levels.

For image modeling, the key distinction is whether the standard defines visually gradable severity. Asphalt crack distress types such as fatigue, block, edge, longitudinal, and transverse cracking have low/moderate/high severity rules. In contrast, FHWA LTPP records asphalt bleeding, polished aggregate, and raveling by affected area and defines no severity level for them. FHWA LTPP also treats PCC map cracking and scaling as surface defects without defined severity levels.

| attribute | structure type | image-assessable? | visual proxy | governing standard |
|---|---|---:|---|---|
| Fatigue/alligator cracking | Asphalt pavement | Yes | Interconnected crack pattern, connectivity, spalling, sealing, pumping evidence; low/moderate/high severity by visible pattern, area recorded in square meters | FHWA LTPP DIM: https://www.fhwa.dot.gov/publications/research/infrastructure/pavements/ltpp/reports/03031/03031.pdf |
| Block cracking | Asphalt pavement | Yes | Block-shaped crack network; FHWA LTPP crack distresses have low/moderate/high severity | FHWA LTPP DIM: https://www.fhwa.dot.gov/publications/research/infrastructure/pavements/ltpp/13092/13092.pdf |
| Longitudinal cracking | Asphalt pavement | Yes | Long cracks aligned with travel direction; FHWA LTPP crack distresses have low/moderate/high severity | FHWA LTPP DIM; TxDOT taxonomy: https://www.txdot.gov/manuals/mnt/pdm/pavement_evaluation/visual_p_cond_surveys-i1004115.html |
| Transverse cracking | Asphalt pavement | Yes | Cross-travel cracks; FHWA LTPP crack distresses have low/moderate/high severity | FHWA LTPP DIM; TxDOT taxonomy: https://www.txdot.gov/manuals/mnt/pdm/pavement_evaluation/visual_p_cond_surveys-i1004115.html |
| Edge cracking | Asphalt pavement | Yes | Cracks near pavement edge; FHWA LTPP crack distresses have low/moderate/high severity | FHWA LTPP DIM: https://www.fhwa.dot.gov/publications/research/infrastructure/pavements/ltpp/13092/13092.pdf |
| Rutting | Asphalt pavement | Partial | Wheel-path depression is visually visible in many images, but depth needs calibrated geometry or survey measurement; TxDOT recognizes shallow/deep/severe rutting | TxDOT visual survey: https://www.txdot.gov/manuals/mnt/pdm/pavement_evaluation/visual_p_cond_surveys-i1004115.html |
| Raveling | Asphalt pavement | Partial | Surface wearing from dislodged aggregate and asphalt binder loss; proxy for aggregate/binder loss and mixture-related performance problems; FHWA LTPP has no severity level and records area | FHWA LTPP DIM: https://www.fhwa.dot.gov/publications/research/infrastructure/pavements/ltpp/13092/13092.pdf |
| Bleeding/flushing | Asphalt pavement | Partial | Dark/shiny binder-rich surface or wheel-path flushing; proxy for excess binder/water-related surface state, but FHWA LTPP has no severity level and records area | FHWA LTPP DIM; TxDOT visual survey: https://www.txdot.gov/manuals/mnt/pdm/pavement_evaluation/visual_p_cond_surveys-i1004115.html |
| Polished aggregate | Asphalt pavement | Partial | Smooth/glossy exposed aggregate texture; FHWA LTPP records area and defines no severity level | FHWA LTPP DIM: https://www.fhwa.dot.gov/publications/research/infrastructure/pavements/ltpp/13092/13092.pdf |
| Asphalt binder content | Asphalt pavement | Lab-only | Single RGB can show raveling or bleeding/flushing proxies, but not binder percentage | Engineering interpretation from FHWA LTPP proxy limits: https://www.fhwa.dot.gov/publications/research/infrastructure/pavements/ltpp/13092/13092.pdf |
| Density / air voids | Asphalt pavement | Lab-only | No reliable single-image proxy; requires core/lab or instrumented testing | Engineering interpretation; no confirmed RGB determinability claim |
| Water damage / stripping progression | Asphalt pavement | Lab-only | Raveling, bleeding/flushing, and polished aggregate are only visual proxies for binder/aggregate loss and water/stripping-related degradation; progression requires time series and/or testing | Engineering interpretation from FHWA LTPP proxy limits: https://www.fhwa.dot.gov/publications/research/infrastructure/pavements/ltpp/13092/13092.pdf |
| Map cracking | PCC pavement/concrete | Partial | Shallow surface-only crack map; FHWA LTPP records occurrence and affected area with no severity level | FHWA LTPP DIM: https://www.fhwa.dot.gov/publications/research/infrastructure/pavements/ltpp/13092/13092.pdf |
| Scaling | PCC pavement/concrete | Partial | Deterioration of upper concrete slab surface, normally 3 to 13 mm deep; visible surface loss is a proxy, but depth needs measurement | FHWA LTPP DIM: https://www.fhwa.dot.gov/publications/research/infrastructure/pavements/ltpp/13092/13092.pdf |
| Spalling | Concrete bridges/slabs | Partial | Broken/delaminated concrete at cracks/edges; severe spall threshold depends on depth or diameter, so single RGB can flag presence but not certify depth | AASHTO element guide: https://apmgs.ro/files/documente/AASHTO-bridge_element_guide_manual__05092010.pdf |
| Efflorescence | Concrete structures | Partial | White crystalline surface deposits can be detected visually; verified findings do not provide a governing severity scale | Engineering interpretation only; no confirmed severity claim |
| Honeycombing | Concrete structures | Partial | Voids/rough exposed matrix can be detected visually; verified findings do not provide a governing severity scale | Engineering interpretation only; no confirmed severity claim |
| Exposed aggregate/rebar | Concrete structures | Partial | Visible aggregate or reinforcement exposure; image can detect exposure, but depth/section loss requires measurement | AASHTO spall thresholds for physical severity: https://apmgs.ro/files/documente/AASHTO-bridge_element_guide_manual__05092010.pdf |

### Image-Only Limits

The verified findings support using images for visible distress presence, crack pattern, affected area proxies, and some ordinal severity labels when the governing standard defines visual severity. They do not support inferring asphalt binder content, density/air voids, or water-damage progression from a single RGB image. Those targets require core samples, lab testing, instrumented survey, GPR, calibrated geometry, or repeated observations. Raveling, bleeding/flushing, and polished aggregate can be learned as visual proxies for binder/aggregate loss, water/stripping-related degradation, and surface texture change, but not as direct material-property measurements.

## B. Official Norms And Rating Scales

### FHWA LTPP Distress Identification Manual

FHWA LTPP defines a common pavement distress taxonomy, severity levels, and measurement methods across three pavement types: asphalt concrete-surfaced pavement, jointed PCC, and continuously reinforced PCC. Distresses are grouped by type, including cracking, surface deformation, surface defects, and miscellaneous distresses. Each distress is assigned a measurement unit such as square meters, millimeters, or number/meters. Only some distresses have severity levels.

For asphalt, fatigue, block, edge, longitudinal, and transverse cracking have low/moderate/high severity. Fatigue severity is visually grounded in crack connectivity, spalling, sealing, and pumping evidence; affected area is recorded at each severity, and mixed areas are rated at the highest severity present. Bleeding, polished aggregate, and raveling have no FHWA LTPP severity level and are recorded by affected area. For PCC, map cracking and scaling are recognized surface defects with no defined severity; scaling is deterioration of the upper slab surface normally 3 to 13 mm deep.

### ASTM D6433 PCI

ASTM D6433-20 defines the Pavement Condition Index method for determining road and parking-lot pavement condition through visual surveys of observed surface distress. PCI is a numerical indicator of pavement surface condition, derived indirectly from observed surface distress, and reflects both structural integrity and functional condition.

**Engineering interpretation:** Tarmac should treat PCI as an aggregate condition target derived from visible distress observations, not as a direct image label unless the training data includes PCI labels or a defensible mapping from observed distresses to an ordinal proxy.

### TxDOT Visual Survey Scale

TxDOT visual pavement condition surveys define flexible-pavement distress types including rutting, patching, failures, block cracking, alligator/fatigue cracking, longitudinal cracking, transverse cracking, raveling, and flushing/bleeding. Rigid-pavement distress types include spalled cracks, punchouts, corner breaks, D-cracking, popouts, and shattered slabs. TxDOT rates raveling and flushing/bleeding qualitatively on a low/medium/high scale; raveling extent is measured as percent of total lane area, and flushing extent as percent of wheel-path length affected. TxDOT also lists rutting as shallow, deep, and severe.

### FHWA SNBI Bridge Component Rating

FHWA SNBI defines bridge component condition rating codes from 0 to 9, plus `N` for not applicable. Each deck, superstructure, substructure, and culvert receives a single overall component rating; deck rating is determined from inspection of all deck surfaces, including top, underside, and edges.

| code | meaning from verified findings |
|---:|---|
| 9 | Excellent: isolated inherent defects |
| 8 | Very good: some inherent defects |
| 7 | Good: some minor defects |
| 6 | Satisfactory: widespread minor or isolated moderate defects |
| 5 | Fair: some moderate defects; strength/performance not affected |
| 4 | Poor: widespread moderate or isolated major defects; strength/performance affected |
| 3 | Serious |
| 2 | Critical |
| 1 | Imminent failure: bridge closed to traffic due to component condition |
| 0 | Failed: bridge closed due to component condition |

### AASHTO Element-Level Four-State Scale

SNBI element-level inspection reports the quantity of each element in Condition State One through Four, separate from component-level 0-9 ratings. AASHTO bridge element inspection uses a standardized four-level scale with general descriptions: good, fair, poor, and severe.

Verified concrete thresholds:

- Crack width: hairline/minor cracks less than 0.0625 in / 1.6 mm map to Condition State 1.
- Crack width: narrow/moderate cracks 0.0625 to 0.125 in / 1.6 to 3.2 mm map to Condition State 2.
- Crack width: cracks greater than 0.125 in / 3.2 mm map to higher condition states.
- Spall severity: severe spall is greater than 1 in / 25 mm deep or greater than 6 in diameter; moderate spall is less than 1 in deep and less than 6 in diameter.

## C. What This Means For Our Model

### Proposed Heads And Standard Mapping

| model output | target labels | standard mapping |
|---|---|---|
| Surface type head | asphalt, PCC/concrete, other project classes | Supports LTPP routing into asphalt, JCP/JPCP, and CRCP/PCC taxonomies |
| Crack type head | fatigue/alligator, block, longitudinal, transverse, edge, none | FHWA LTPP asphalt crack distress taxonomy; TxDOT flexible-pavement taxonomy |
| Crack severity head | low, moderate, high | FHWA LTPP visual severity for asphalt crack distresses |
| Surface defect head | raveling, bleeding/flushing, polished aggregate, map cracking, scaling | Presence/area proxy only for FHWA LTPP no-severity defects; TxDOT can provide low/medium/high for raveling/flushing if labeled accordingly |
| Rutting head | none/shallow/deep/severe or scalar depth when geometry is available | TxDOT visual rutting terms; depth should require calibrated geometry |
| Concrete element defect head | cracking, spalling, exposed aggregate/rebar, surface defect | AASHTO/FHWA bridge element inspection concepts |
| Bridge element condition head | Condition State 1-4 | AASHTO element-level good/fair/poor/severe scale |
| Bridge component condition head | 0-9 or coarser ordinal bins | FHWA SNBI component condition scale |
| Overall condition grade | 1-5 project grade | PCI-like ordinal proxy, not official PCI unless derived from full ASTM D6433 survey inputs |

### Insufficient From Single RGB

Do not train or expose single-image predictions as direct measurements of asphalt binder content, density/air voids, stripping progression, or water-damage progression. Use metadata, multi-view imagery, calibrated scale, geometry/depth sensors, GPR, cores/lab results, and time-series observations when those targets matter. Single RGB can provide visible proxies: raveling as aggregate/binder loss, bleeding/flushing as binder-rich surface, polished aggregate as texture loss, and increasing crack/spall area as deterioration signals.
