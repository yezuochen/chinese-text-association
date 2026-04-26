# Directional and Causal Scoring Between Concepts in Time-Indexed Text Corpora
## A Survey of Algorithms, Indices, and Methods for Financial NLP Applications

This report surveys the concrete, scalar-valued methods that produce **asymmetric / directional scores** between concepts or words — the kind of quantity one would need to infer claims such as *"inflation tends to precede / cause recession"* from a timestamped corpus of financial news. It covers (1) directional co-occurrence/association measures, (2) causal relation extraction systems that output scalar causal scores, (3) document-level causal graph learners for finance (including Granger-causality applied to text time series), and (4) asymmetric / order-based embedding geometries. For each method I report the level at which it operates (word vs. sentence vs. document-corpus), its output, and any explicit application to finance / economics.

---

## 1. Directional Association Measures (Word- or Token-Level, Asymmetric Scalars)

### 1.1 ΔP (Delta-P) and directional transitional probabilities
ΔP is the canonical *directional* collocation measure. For two words w₁, w₂ it returns two different values: ΔP(w₂|w₁) = P(w₂|w₁) − P(w₂|¬w₁) and ΔP(w₁|w₂) = P(w₁|w₂) − P(w₁|¬w₂). The asymmetry matches the psycholinguistic intuition that *mango → fruit* is a strong forward association while *fruit → mango* is weak — exactly the primitive one needs to quantify "A tends to appear after B." The measure is bounded in [-1, 1] and has been shown to outperform or rival symmetric association statistics in predicting sequencing phenomena such as hesitation placement ([Schneider, ΔP as a measure of collocation strength, CLLT 2018](https://www.degruyterbrill.com/document/doi/10.1515/cllt-2017-0036/html?lang=en)) and in corpus-linguistics reviews that make the directionality explicit ([Gries, 50-something years of work on collocations, IJCL 2013](https://stgries.info/research/2013_STG_DeltaP&H_IJCL.pdf)). Dunn extended ΔP to arbitrary-length directional multi-unit measures ([Dunn, Multi-Unit Directional Measures of Association, arXiv:2104.01297](https://arxiv.org/abs/2104.01297)). Because ΔP is computed from simple conditional probabilities over a sequential window, it is immediately applicable to ordered token streams in financial news (e.g., using "t < t+k" windows to measure whether "inflation" elevates the conditional probability of "recession" appearing later).

### 1.2 Asymmetric PMI variants
Directional PMI replaces the symmetric joint in P(x,y)/(P(x)P(y)) with an ordered one. Michelbacher, Evert & Schütze explicitly showed that "virtually all association measures are symmetric… native speakers have strong intuitions that in many cases one term in a collocation is more 'important' for the other than vice versa," and constructed two asymmetric measures (rank-based and conditional-probability-based) that correctly predict the direction of association in ~80% of free-association pairs ([Michelbacher et al. 2007, Asymmetric Association Measures](https://www.stephanie-evert.de/PUB/MichelbacherEtc2007.pdf); [Michelbacher, Evert & Schütze, 2011](https://aclanthology.org/P14-2049.pdf)). Tools such as Pmizer implement PMI, NPMI, PMI², NPMI², cPMI and support explicit "forward-looking" asymmetric context windows, i.e. counting only co-occurrences where w₂ follows w₁ ([Pmizer GitHub](https://github.com/asahala/Pmizer)). Using an *ordered* (forward-only) window turns PMI into a directional score — PMI(A → B) ≠ PMI(B → A) — which is the simplest asymmetric-scalar baseline for "A precedes B" in a time-respecting text window.

### 1.3 Other directional / asymmetric index families
- **Somers' D** (ordinal contingency): a classic asymmetric measure of association used where one variable is treated as independent, one as dependent ([pressbooks.ric.edu — Measures of Association](https://pressbooks.ric.edu/socialdataanalysis/chapter/an-in-depth-look-at-measures-of-association/)).
- **α-skew divergence** (Lee 1999) and **KL-divergence** between word context distributions: inherently asymmetric; used as prototypical directional word-similarity measures ([Improving sparse word similarity models with asymmetric measures, ACL 2014](https://aclanthology.org/P14-2049.pdf); [Mohammad & Hirst 2012 survey, arXiv:1203.1858](https://arxiv.org/pdf/1203.1858)).
- **Tversky's ratio model** and the **Simplified Asymmetric InfoSimba (AISs)** — an information-theoretic asymmetric measure explicitly designed for *entailment by generality* ([Pais, Asymmetric Distributional Similarity Measures, 2013 PhD thesis](https://pastel.hal.science/pastel-00962176)).

These measures operate at **word (or token) level** and, with the exception of Izumi-Lab's transfer-entropy applications (§3), have not been used directly as causal indices in mainstream financial NLP — but they are the natural primitives.

---

## 2. Entailment-Inspired Asymmetric Distributional Measures

A parallel line of work, motivated by lexical entailment ("narrow term → general term"), has produced several directional scalar indices that compare the *contexts* of two words and return how much one is "subsumed" by the other.

| Measure | Output | Key property | Reference |
|---|---|---|---|
| **WeedsPrec** (Weeds & Weir 2003) | asymmetric precision ∈ [0,1] | proportion of u's context mass contained in v's contexts | [Weeds & Weir 2003; used in entailment graphs](https://aclanthology.org/Q18-1048.pdf) |
| **ClarkeDE** (Clarke 2009) | asymmetric inclusion ∈ [0,1] | degree-of-entailment based on min of components | [Kartsaklis & Sadrzadeh review](https://aclanthology.org/C16-1268.pdf) |
| **APinc / balAPinc** (Kotlerman et al. 2010) | directional "distributional-inclusion" AP score | averaged-precision weighting of shared features, balanced with Lin's symmetric similarity | [Kotlerman et al., Natural Language Engineering 2010](https://www.cambridge.org/core/journals/natural-language-engineering/article/abs/directional-distributional-similarity-for-lexical-inference/666EA64B78F0A559FA3B1FC5DAA7B054) |
| **invCL** (Lenci & Benotto 2012) | directional inclusion/exclusion score | combines distributional inclusion of u in v with non-inclusion of v in u | [Lenci & Benotto 2012](https://aclanthology.org/P16-1226.pdf) |
| **SLQS** (Santus et al. 2014) | entropy-based directional score | uses H(c) of each word's most-associated contexts to identify the more "general" word | [Santus et al., EACL 2014](https://aclanthology.org/E14-4008.pdf); [Improving Hypernymy Detection, 2016](https://arxiv.org/pdf/1603.06076) |

These measures are defined on **word-level** distributional vectors and all return a single directional scalar. They have been the state of the art for *unsupervised hypernymy directionality* for years; while originally motivated by taxonomic relations, several authors (Schlechtweg, Kotlerman) note that SLQS is also "independently measurable over time, which avoids the problem of vector space alignment" — making it a candidate for diachronic/directional financial analysis ([Demasking Unsupervised Hypernymy Prediction Methods](https://www.researchgate.net/publication/352054349_More_than_just_Frequency_Demasking_Unsupervised_Hypernymy_Prediction_Methods)). None has been widely applied to finance, but all are directly portable.

---

## 3. Granger Causality Applied to Word / Concept Time Series from News

This is the most developed thread of work that directly answers "A Granger-causes B" at the concept level, from timestamped documents.

### 3.1 Balashankar et al. — Predictive causal graphs from news
Balashankar, Jagabathula & Subramanian (AAAI 2019 and ICML 2023) pioneered building **predictive causal graphs** in which edges are Granger-causal relations between daily frequencies of event/term mentions in the New York Times. Their follow-up, *Learning Conditional Granger Causal Temporal Networks* ([Balashankar et al., 2023, PMLR v213](https://proceedings.mlr.press/v213/balashankar23a.html); [PDF](https://ananthbalashankar.github.io/granger_causal.pdf)), reports a 25% improvement in area under the precision-recall curve for Granger-causal link discovery and 18–25% better forecasting on NYT-derived stock-price prediction, explicitly combining a news-term time series with stock returns.

### 3.2 Maisonnave et al. — Causal graphs from news (NYT)
[Maisonnave et al. (PeerJ CS 2022)](https://peerj.com/articles/cs-1066/) and [the companion PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC9374167/) propose a framework that builds a time series per term and per *ongoing* event trigger, then compares four time-series causal-discovery methods (including classic Granger, PCMCI, and other constraint-based approaches) on a large NYT corpus. A notable point is that this enables detection of **implicit causal links between words that never appear in the same article**, because the signal is derived from synchronous changes in document-corpus frequencies rather than sentence co-occurrence.

### 3.3 Framing & agenda-setting (Russian news)
[Field et al., Framing and Agenda-setting in Russian News, 2018](https://www-nlp.stanford.edu/pubs/field2018framing.pdf) show a concrete workflow: monthly term-frequency time series from the Izvestia corpus are tested with standard Granger-causality (VAR + F-test) against the RTSI stock index, and the sign and p-value define the directional score. A p ≤ 0.05 test at lag 1 finds RTSI values Granger-causing U.S.-topic coverage.

### 3.4 Neural, non-linear and variable-lag Granger methods
- **Neural Granger Causality** (Tank, Covert, Foti, Shojaie, Fox — TPAMI 2022) uses component-wise MLP / LSTM networks with group-lasso penalties on input weights; a zeroed input-weight group means "series j does not Granger-cause series i." Output is a sparse directional adjacency matrix. Reference implementation [iancovert/Neural-GC](https://github.com/iancovert/Neural-GC); paper [arXiv:1802.05842](https://arxiv.org/abs/1802.05842). Applied to DREAM3 and MoCap; directly applicable to document-level concept time series.
- **Variable-Lag Granger Causality** and its *Multi-Band* (MB-VLGC) extension permit delays that vary over time and by frequency band, relevant for causal delays between news topics that evolve non-stationarily ([arXiv:2508.00658](https://arxiv.org/html/2508.00658)).
- **Transfer Entropy** is a non-parametric, model-free directional generalization of Granger causality and reduces to it for linear VAR processes ([Wikipedia: Transfer Entropy](https://en.wikipedia.org/wiki/Transfer_entropy)); Behrendt et al.'s *RTransferEntropy* is a mature R package with financial-time-series examples ([ScienceDirect, RTransferEntropy](https://www.sciencedirect.com/science/article/pii/S2352711019300779)). Izumi, Suzuki & Toriumi applied transfer entropy to measure information flow between financial-news-derived variables and stock-market quantities ("Transfer Entropy Analysis of Information Flow in a Stock Market," Springer 2017; listed in [Izumi-Lab publications](https://sites.google.com/socsim.org/izumi-lab/publications/paper)). TE is, to my knowledge, the cleanest non-parametric, asymmetric, *time-respecting* scalar currently used in financial NLP settings.

### 3.5 CSHT — Granger-causal hypergraph transformer
The very recent *Causal Sphere Hypergraph Transformer (CSHT)* encodes multivariate Granger-causal dependencies between financial news/sentiment features and asset returns as directional hyperedges on a sphere, then uses a causally-masked Transformer to forecast returns ([arXiv:2510.04357](https://arxiv.org/html/2510.04357)). It is an explicit application of directional, Granger-based scoring between news concepts and financial targets.

### 3.6 Sentiment-concept Granger tests in finance (caveats)
Many papers (e.g., [Souma et al., PMC 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9815756/); [Mbonu et al., Sentimental Showdown, PMC 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11076966/); [Forecasting with Economic News, Taylor & Francis 2022](https://www.tandfonline.com/doi/full/10.1080/07350015.2022.2060988)) apply Granger-causality tests between *aggregate sentiment* extracted via NLP and market returns, finding mixed, often non-significant results (FTSE study rejected news-sentiment → return causality). These work at the document/concept-aggregate level and illustrate that the Granger paradigm is standard practice; the null results also caution that raw term-frequency Granger scores need bias correction (non-stationarity, low sample size, multiple testing).

---

## 4. Causal Relation Extraction at Sentence / Document Level (Scalar Causal Scores)

These systems produce a *pairwise causal score* between two spans (events, concepts) by combining linguistic patterns, discourse markers, and neural classifiers.

### 4.1 Pattern and distributional scoring
- **Girju (2003)** — lexico-syntactic patterns for cause–effect detection ([Automatic Detection of Causal Relations, ACL 2003 workshop](https://aclanthology.org/P03-1054)).
- **Do, Chan & Roth (EMNLP 2011), "Minimally Supervised Event Causality Identification"** — combines *Cause-Effect Association (CEA)* via PMI over pattern pairs with discourse connective features; the CEA score is an asymmetric scalar over (event1, event2). Paper [EMNLP 2011](https://aclanthology.org/D11-1027/). This is often cited as the closest off-the-shelf "causal-PMI" index.
- **Beamer & Girju (2009)** — bigram event model with "causal potential."
- **Riaz & Girju (2013, 2014)** — verb-verb and verb-noun causal-power associations.
- **Hashimoto et al. (EMNLP 2012), Excitatory-or-Inhibitory** — introduces a new *semantic orientation* (E/I) that marks template arguments as activated (excitatory, +) or suppressed (inhibitory, −) by the predicate, producing a signed directional score; score range [−1, 1] where positive = excitatory template, negative = inhibitory; used successfully to extract contradiction and causality from the web ([Hashimoto et al., EMNLP 2012](https://aclanthology.org/D12-1057/)). This is particularly interesting for finance: "tightening → inflation" vs. "loosening → inflation" would carry opposite E/I scores.

### 4.2 Neural causal scorers (produce a scalar P(causal))
- **Causal-BERT / CausalBERT** — Khetan et al. fine-tune BERT variants on BECauSE/SemEval/EventStoryLine to return a scalar causal probability between two event mentions ([Khetan, Ramnani et al., "Causal BERT," Springer 2021](https://link.springer.com/chapter/10.1007/978-3-030-80119-9_64); [Li & Ding, CausalBERT: injecting causal knowledge into pre-trained models](https://www.semanticscholar.org/paper/CausalBERT%3A-Injecting-Causal-Knowledge-Into-Models-Li-Ding/ff2f48fe6438adcaf860aac0f41c584568beafb5)).
- **Dasgupta et al. (SIGdial 2018)** — Bi-LSTM + linguistic features for cause-effect extraction, outputting sentence-level tagged spans and implicit scalar classification; also used to build a causal graph ([ACL Anthology](https://aclanthology.org/W18-5035/)).
- **BioBERT-BiGRU** and variants for causal relation transfer learning ([arXiv:2503.06076](https://arxiv.org/html/2503.06076)).
- **Event Causality Identification (ECI)** has become a dedicated sub-task with a 2025 ACM Computing Surveys taxonomy that explicitly notes "this capability is particularly essential in domains that require complex reasoning, such as finance" ([ACM CSUR, 2025](https://dl.acm.org/doi/10.1145/3756009)).
- **Event Causality Identification with Synthetic Control** (Wang, Liu, Zhang, Roth, Richardson — EMNLP 2024) adopts the Rubin causal model over temporally ordered events and uses synthetic-control from embeddings to estimate pairwise causal likelihoods; it outperforms GPT-4 on COPES-hard ([EMNLP 2024 ACL Anthology](https://aclanthology.org/2024.emnlp-main.103/)).
- **Sharp, Surdeanu, Jansen, Clark, Hammond (EMNLP 2016), "Creating Causal Embeddings for QA with Minimal Supervision"** — trains dedicated *causal embeddings* where the context of a cause is its effect; produces a directional similarity score optimized for causality rather than general relatedness ([ACL Anthology D16-1014](https://aclanthology.org/D16-1014/)). This is arguably the first "causal embedding" formally named as such in NLP.

### 4.3 Benchmark corpora (define the scalar output)
- **BECauSE 2.0** (Dunietz, Levin, Carbonell, LAW 2017) — exhaustively annotated cause/effect with seven overlapping semantic relations; basis of most English supervised scorers ([LAW 2017](https://aclanthology.org/W17-0812/); [GitHub](https://github.com/duncanka/BECauSE)).
- **EventCausality** (Do et al. 2011) — 485 sentences.
- **Event StoryLine Corpus** (Caselli & Vossen 2017) — 2,608 pairs with causal + temporal links.
- **Causal-TimeBank** (Mirza et al. 2014), **CaTeRS** (Mostafazadeh 2016), **COPA**, **PDTB 3.0** — all compiled via **CREST** into a unified directional-causal schema (Hosseini, Broniatowski & Diab, [arXiv:2103.13606](https://arxiv.org/abs/2103.13606); [GitHub](https://github.com/phosseini/CREST)). CREST provides a "direction" field (span1 → span2 / span2 → span1).
- **ESTER** (Han et al., EMNLP 2021) — 10.1K event-relation pairs including causal, sub-event, conditional, counterfactual reasoning, via MRC queries ([arXiv:2104.08350](https://arxiv.org/abs/2104.08350)).
- **CRAB** (Romanou et al., EMNLP 2023) — *strength* of causal relations between real-world events, producing graded scalar causal scores ([in CausalNLP_Papers list](https://github.com/zhijing-jin/CausalNLP_Papers)).

### 4.4 Large-scale causal knowledge bases (each edge = scalar causal score)
- **CauseNet** (Heindorf, Scholten, Wachsmuth, Ngonga Ngomo, Potthast — CIKM 2020) — 11,609,890 causal relations between causal concepts mined from Wikipedia + ClueWeb12 with estimated extraction precision of 83% ([CauseNet website](https://causenet.org/); [paper PDF](https://downloads.webis.de/publications/papers/heindorf_2020.pdf); [CIKM dblp entry](https://dblp.org/rec/conf/cikm/HeindorfSWNP20.html)). Edges weighted by corroborating evidence frequency.
- **IBM Causal Knowledge Extraction** (Hassanzadeh, Bhattacharjya, Feblowitz, Srinivas, Perrone, Sohrabi, Katz — AAAI 2020 demo) — large-scale news mining that returns an API-level score of *likelihood of causal relation* given two phrases, for enterprise-risk-management use ([AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/7092); [IBM Research](https://research.ibm.com/publications/causal-knowledge-extraction-through-large-scale-text-mining)). An IJCAI 2019 companion evaluated unsupervised methods on human-expert cause-effect pairs ([IBM Research](https://research.ibm.com/publications/answering-binary-causal-questions-through-large-scale-text-mining-an-evaluation-using-cause-effect-pairs-from-human-experts)). This is arguably the closest thing to a production-grade financial-domain causal scorer.
- **Pundit / Radinsky et al., WWW 2012** — early news-based causality predictor that trained on 150 years of NYT headlines and mined ~30M fact nodes with >1B edges; each edge is a generalized causal rule with confidence ([WWW 2012](https://dl.acm.org/doi/10.1145/2187836.2187958); [arXiv:1402.0574](https://arxiv.org/abs/1402.0574); [Radinsky thesis / Technion PDF](https://csaws.cs.technion.ac.il/~shaulm/papers/pdf/Radinsky-Davidovich-Markovitch-WWW2012.pdf)). Inter-annotator averaged precision ≈ 78%, recall 10%.

---

## 5. Finance-Specific Causal NLP

### 5.1 FinCausal shared task (2020 → 2025)
The **FinCausal** series (Mariko, Abi-Akl, Labidurie, Durfort, de Mazancourt, El-Haj, Moreno-Sandoval and colleagues) is the flagship benchmark for causality detection in financial documents, run annually at FNP/FNP-FNS since 2020:
- **FinCausal 2020** — binary causality classification + cause/effect span extraction on SEC/Edgar and Qwam news ([ACL Anthology](https://aclanthology.org/2020.fnp-1.3/); [arXiv:2012.02505](https://arxiv.org/abs/2012.02505)).
- **FinCausal 2021 / 2022** — refined tasks focused on quantified facts; top F1 ≈ 95.5% ([ACL 2022.fnp-1.16](https://aclanthology.org/2022.fnp-1.16/)).
- **FinCausal 2023** — English + Spanish subtasks, best F1 0.54 from ChatGPT+CoT prompts ([ResearchGate](https://www.researchgate.net/publication/376678468_The_Financial_Document_Causality_Detection_Shared_Task_FinCausal_2023); [arXiv: LTRC_IIITH submission](https://arxiv.org/html/2401.13545)).
- **FinCausal 2025** — multilingual (English + Spanish) causality extraction from annual reports ([ACL 2025.finnlp-1.21](https://aclanthology.org/2025.finnlp-1.21/)).

These produce per-sentence causal pair extractions (not corpus-level scores) but provide the training data that downstream corpus-wide aggregators rely on.

### 5.2 Japanese Izumi-Lab line (economic causal chains)
- **Sakaji, Sekine & Masuyama (2008)** — rule-based extraction with clue phrases from Nikkei newspapers.
- **Sakaji, Murono, Sakai, Bennett, Izumi (CIFEr 2017)** — rare causal knowledge mining from financial statement summaries.
- **Izumi & Sakaji (FinNLP 2019), "Economic Causal-Chain Search"** — three-step pipeline (causal-sentence SVM → pattern extraction → chain construction via word2vec similarity); deployed as an economic news-impact engine ([Springer chapter](https://link.springer.com/chapter/10.1007/978-3-030-56150-5_2); [ACL W19-5510 PDF](https://aclanthology.org/W19-5510.pdf)).
- **Sakaji & Izumi (New Generation Computing, 2023)** — extends to bi-lingual (Japanese + English) financial causality extraction via Universal Dependencies ([Springer](https://link.springer.com/article/10.1007/s00354-023-00233-2)).
- **Nakagawa, Sashida, Sakaji, Izumi (2019)** — *Economic Causal Chain and Predictable Stock Returns*: uses the causal chain as a lead-lag predictor for returns.

### 5.3 FinCaKG-Onto — Financial Causal Knowledge Graph with ontology
[Wang, Izumi & Sakaji (Applied Intelligence, 2025), FinCaKG-Onto](https://link.springer.com/article/10.1007/s10489-025-06247-1), weights edges by "the frequency of causality occurrences within our corpus," providing per-concept-pair strength scores for financial causal reasoning.

### 5.4 FinDKG, BloombergGPT and financial knowledge graphs
- **FinDKG** (Li & Sanna Passino, ICAIF 2024) — a dynamic, time-stamped financial knowledge graph built by a fine-tuned LLM (ICKG) over ~400K WSJ articles 1999–2023. Quadruples (s, r, o, t) carry temporal resolution, and a KGTransformer GNN predicts future links, achieving ~15% MRR improvement over prior TKG baselines and outperforming thematic ETFs in a back-test ([arXiv:2407.10909](https://arxiv.org/abs/2407.10909); [project site](https://xiaohui-victor-li.github.io/FinDKG/); [GitHub](https://github.com/xiaohui-victor-li/FinDKG)). Causal edges are among the relation types.
- **BloombergGPT** (Wu, İrsoy, Lu, Dabravolski, Dredze, Gehrmann, Kambadur, Rosenberg, Mann — 2023) — a 50-billion-parameter decoder-only **causal language model** (the term "causal" here means autoregressive LM masking, *not* causal inference) trained on 363B tokens of financial text plus 345B public tokens, used for sentiment, NER, classification and QA on Bloomberg data ([arXiv:2303.17564](https://arxiv.org/abs/2303.17564); [Bloomberg press release](https://www.bloomberg.com/company/press/bloomberggpt-50-billion-parameter-llm-tuned-finance/)). It does not itself output directional causal scores between concepts, but it is the usual underlying encoder for downstream causal-scoring heads in finance.
- **Tan, Paul, Yamaura, Miura, Ng (2023), "Constructing and Interpreting Causal Knowledge Graphs from News"** — two-step pipeline (extract causal relations, then cluster and represent arguments) that explicitly targets recall + precision + interpretability for financial news ([arXiv:2305.09359](https://arxiv.org/pdf/2305.09359)).
- **FinCARE / FinReflectKG** — hybrid statistical + LLM causal graph construction from SEC 10-K filings; reports F1 = 0.759 for DAG recovery using NOTEARS with KG + LLM priors ([arXiv:2510.20221](https://arxiv.org/html/2510.20221v1)).

### 5.5 News → return predictive causal graphs
Beyond Balashankar et al., several 2023–2025 papers build directed concept→stock graphs:
- **TRACE** (arXiv 2603.12500) — rule-guided reasoning over a temporal KG of S&P 500 news with 174K edges, including causal motifs.
- **Causal Sphere Hypergraph Transformer** (arXiv 2510.04357) — Granger-causal hyperedges on the sphere for news-sentiment → returns.
- **Implicit-Causality-Exploration-Enabled GNN** (MDPI *Information* 2024) — uses Granger causality explicitly for stock-prediction graph construction ([MDPI](https://www.mdpi.com/2078-2489/15/12/743)).

---

## 6. Asymmetric / Directional Embedding Geometries

These methods do not use sequential position per se, but they produce a **directional scalar** between two words or concepts whose magnitude encodes generality/specificity or entailment — and are frequently combined with corpus training on time-indexed data.

### 6.1 Order Embeddings (Vendrov, Kiros, Fidler, Urtasun, ICLR 2016)
Represents a concept as a point in ℝⁿ₊ with the partial order "x ≤ y ⟺ x_i ≤ y_i ∀i." Ordering encodes hypernymy / entailment as a directional inequality. Output is a non-negative asymmetric score E(x, y) = ‖max(0, y − x)‖² ([arXiv:1511.06361](https://arxiv.org/abs/1511.06361)).

### 6.2 Poincaré / Hyperbolic embeddings (Nickel & Kiela, NIPS 2017)
Embeds symbolic data in a Poincaré ball so that distance encodes similarity *and* norm encodes hierarchy level (generality ↓ as norm ↑) ([NeurIPS 2017](https://papers.nips.cc/paper/7213-poincare-embeddings-for-learning-hierarchical-representations); [arXiv:1705.08039](https://arxiv.org/abs/1705.08039)). The Lorentz-model variant (Nickel & Kiela, ICML 2018) improves optimisation. Output is a directional hierarchy-aware scalar derivable from distance + norm difference.

### 6.3 Hyperbolic Entailment Cones (Ganea, Bécigneul, Hofmann, ICML 2018)
Embeds DAGs / partial orders as *nested geodesically-convex cones*; membership provides a sharply asymmetric entailment score with closed-form optimum in both Euclidean and hyperbolic geometries ([ICML 2018 PMLR](https://proceedings.mlr.press/v80/ganea18a.html); [arXiv:1804.01882](https://arxiv.org/abs/1804.01882); [GitHub](https://github.com/dalab/hyperbolic_cones)).

### 6.4 Gaussian Embeddings (Vilnis & McCallum, ICLR 2015)
Each word is a Gaussian N(μ, Σ); asymmetric similarity is the KL-divergence KL(N_w₁ ‖ N_w₂), naturally directional; variance encodes specificity/ambiguity ([arXiv:1412.6623](https://arxiv.org/abs/1412.6623); reference implementation [word2gauss](https://github.com/seomoz/word2gauss)). Density Order Embeddings (Athiwaratkun & Wilson, 2018) extend this with explicit encapsulation losses.

### 6.5 Box-lattice / Probabilistic Box Embeddings (Vilnis, Li, Murty, McCallum, ACL 2018)
Each concept is an axis-aligned hyperrectangle. The score *P(y | x) = Vol(x ∩ y) / Vol(x)* is a proper directional conditional probability, exactly the form one needs for "given that inflation occurs in a document, what is the probability of recession?" Later "smoothing" and Gumbel-box variants improve gradients ([ACL 2018 anthology](https://aclanthology.org/P18-1025/); [arXiv:1805.06627](https://arxiv.org/abs/1805.06627)). Box embeddings have been adopted for knowledge-graph completion and can be trained from co-occurrence/temporal-conditioning statistics.

### 6.6 HyperVec, Hierarchical Density Order Embeddings, Probabilistic Order Embeddings (Lai & Hockenmaier 2017) — all produce directional scalars and build on these core ideas.

For finance specifically, these geometries have been used in thematic ways (FinDKG uses knowledge-graph embeddings over financial entities; HyDEN etc.) but direct applications of Poincaré / box / order embeddings to a time-stamped financial news corpus for causal-ordering inference remain uncommon — they are available building blocks.

---

## 7. Diachronic / Time-Indexed Word Embeddings

While not directly causal, these methods produce trajectories of word vectors across time slices and can feed downstream directional-scoring (e.g., detect that "recession" neighbourhood shifts *after* "inflation" spikes).

- **Hamilton, Leskovec & Jurafsky (ACL 2016), "Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change"** — Procrustes-aligned SVD/SGNS embeddings per time slice; rate-of-change as a per-word scalar.
- Kutuzov, Øvrelid, Szymanski, Velldal (COLING 2018 survey, [ACL](https://aclanthology.org/C18-1117.pdf)) catalogue the field.
- Rosin, Radinsky & Adar — *Temporal word analogy*: models relations of form "X in year t₁ is to Y in year t₂" as directional vectors, giving an explicit time-indexed directional score between concepts ([cited in Kutuzov thesis](https://www.mn.uio.no/ifi/forskning/aktuelt/arrangementer/disputaser/2020/phd_thesis_kutuzov.pdf)).

These are essential when the *same* concept's meaning shifts (e.g., "cloud," "post") but are less central when the concepts are stable (inflation, recession) — in which case raw term-frequency time series (§3) suffice.

---

## 8. LLMs as Causal Scorers (recent, important caveats)

Kıcıman, Ness, Sharma & Tan ("Causal Reasoning and Large Language Models," [arXiv:2305.00050](https://arxiv.org/abs/2305.00050)) report GPT-4 reaches 97% weighted accuracy on the Tübingen pairwise causal-discovery benchmark (vs. 83% prior best) and 92% on counterfactual reasoning; this suggests LLMs can score directional P(effect | cause) for concept pairs via prompting with concepts only ("no data"). However, [Jin et al.'s Corr2Cause benchmark](https://arxiv.org/html/2505.18034v1) reports GPT-4 at F1 ≈ 29 on *correlation-to-causation* deduction — barely above random — and [a 2025 arXiv critique](https://arxiv.org/html/2506.00844v1) shows systematic sensitivity to word-order in prompts. LLMs can be used as causal scoring oracles over financial-news-derived concept pairs but with substantial caveats about construct validity and prompt-engineering leakage. A dedicated fine-tuning approach, **LLM4Causal** ([arXiv:2312.17122](https://arxiv.org/html/2312.17122v3)), and Sakaji-Lab's *CausalEnhance* (Knowledge-Based Systems, 2025) explicitly target causal text mining with open-source LLMs.

---

## 9. Summary Matrix

| Method family | Level | Output | Time-indexed? | Applied to finance? |
|---|---|---|---|---|
| Asymmetric PMI / forward-window PMI | word | scalar asymmetric | via ordered window | Izumi/Sakaji, Pmizer; limited |
| ΔP (delta-P) | word | scalar ∈ [−1, 1], directional | sequential | no established finance use |
| Directional distributional inclusion (WeedsPrec, balAPinc, ClarkeDE, invCL, SLQS, AISs) | word | scalar asymmetric | static | no |
| Hashimoto E/I orientation | word/template | signed score ∈ [−1, 1] | no | no (but ideal fit for monetary-policy language) |
| Causal-relation extractors (Do/Roth CEA, BECauSE, Causal-BERT, CauseNet, IBM Causal KE, Pundit, Sharp causal embeddings) | sentence / doc-pair | P(causal) or ranked score | no | IBM (risk mgmt), Pundit (NYT), FinCausal |
| FinCausal / FinCaKG-Onto / Izumi economic causal chain / Tan et al. 2023 / FinCARE | sentence → corpus KG | edge weights, scalar cause-effect | per-document timestamp | **yes (finance)** |
| Granger causality on term/event time series (Balashankar, Maisonnave, Field, Tank Neural-GC) | document corpus | directed adjacency + p-values / F-stats | **yes** | **yes (NYT → stocks, macro news → indices)** |
| Transfer entropy on text-derived series (Izumi et al.) | document corpus | non-parametric directional info-flow | **yes** | **yes (stock markets)** |
| CSHT / TRACE / FinDKG / Implicit-causality GNN / BloombergGPT | doc / KG | temporal causal hyperedges, return-predictive | **yes** | **yes** |
| Order / Poincaré / Hyperbolic-cone / Gaussian / Box embeddings | word/concept | asymmetric energy or conditional probability | static (unless trained diachronically) | rare direct use |
| Diachronic word embeddings + temporal word analogies | word | trajectory distances, time-varying analogies | **yes** | limited |
| LLM-as-causal-scorer (Kıcıman et al., LLM4Causal) | concept pair | P(A→B) from prompt | via prompt scaffolding | emerging |

---

## 10. Practical Recommendations for the "Inflation precedes / causes Recession" Task

For a timestamped corpus of financial news, the following produce concrete, asymmetric scalars without relying on raw co-occurrence counts or embedding cosine:

1. **Per-concept daily/weekly frequency time series**, then apply **Granger causality F-tests** and/or **transfer entropy** (explicitly time-directional). Use the Balashankar et al. or Maisonnave et al. pipeline as template. Report p-values and effect sizes per direction as your primary directional scalar.
2. **Forward-window ΔP and directional PMI** over ordered token streams (optionally within-article or within-time-bucket) as a linguistic, low-resource complement.
3. **Train or apply an ECI model (Causal-BERT, FinCausal-style BERT/RoBERTa, or IBM's extractor) at the sentence level**, aggregate scalar causal-link counts or log-likelihoods between concept types to obtain a corpus-level directed graph.
4. **Use an LLM (GPT-4 / LLM4Causal / CausalEnhance) as a zero-shot oracle for P(effect | cause)** on concept pairs and cross-check against the frequency-time-series test; divergences flag spurious linguistic-pattern artifacts.
5. **Optionally embed the resulting concept graph in a Poincaré ball or box-lattice space** to obtain a smooth directional score that respects transitivity and enables downstream querying (FinDKG's KGTransformer is a production-ready reference).

Cross-validating directional scores from independent families (sequential ΔP/PMI, sentence-level BERT causal scorer, aggregate Granger/transfer-entropy on time series, LLM prompting) is the emerging best practice, because each is vulnerable to distinct failure modes (pattern sparsity, confounding, non-stationarity, prompt bias). The FinCausal task, IBM's causal extractor, Izumi-Lab's economic causal-chain system, CauseNet, and FinDKG collectively form the most mature infrastructure for financial-domain directional concept-scoring today.