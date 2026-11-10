# Related work: identity leakage in BCI outputs

This review supports the BEAT technical report (`README.md`, `docs/analysis_plan.md`). It covers
EEG identity protection, critiques of adversarial attribute removal, EEG identifiability across
sessions, privacy mechanisms for EEG/BCI data, and covariance alignment methods that reduce identity
as a side effect.

**Verification basis.** Every statement about a source comes from its opened full text (PDF or
HTML), unless the entry says only the abstract or record page was read. Sources [1]–[20] were read on
2026-11-10. Sources [21]–[36] were added on 2026-11-11 from the abstract or record page only
(Section 7). Sources [37] (the Yang2025 dataset of study M) and [38] (the timestamp protocol of
seal 4) were added later on 2026-11-11 from their Crossref, figshare and RFC Editor records. The
bibliographic details of every entry were checked against Crossref and the arXiv API, and every
DOI, arXiv identifier and URL was resolved by `python scripts/check_references.py`, whose output is
recorded in `results/reference_check/` (Appendix). The labels used are: *peer-reviewed* (journal or proceedings version), *workshop paper*,
*preprint* (arXiv, not peer-reviewed), *unreviewed repository* (GitHub technical report), and *not
opened* (the source could not be retrieved, so no claims are drawn from it). Numbers are quoted as
the sources report them. Different papers use different metrics, attackers and splits, so values in
different rows are **not directly comparable**.

## 1. EEG identity protection and anonymization

**Identity-unlearnable perturbations (Meng et al., 2023).** The method adds sample-wise or user-wise
perturbations to EEG trials so that a user classifier trained on the released data fails to learn
identity, while a task classifier still trains normally. Evaluation uses leave-one-session-out
cross-validation on seven datasets from five paradigms, including BCI Competition IV 2a. A User-Classifier is
retrained on the perturbed training session and tested on the remaining sessions. On average, user
identification accuracy falls from 70.01% to at most 21.36%. The perturbations were generated with
EEGNet and transferred to DeepCNN, ShallowCNN and an LSTM. The authors note that transfer works less well to
architectures that differ more from EEGNet [1].

**ID-RemovalNet (Wang et al., 2025).** The network combines an identity-decorrelation separation
module, a gradient reversal layer, attention-based adversarial feature selection, task-feature
re-fusion and an EEG alignment step. The identifiers are two fully connected layers attached to each
backbone (EEGNet, ShallowCNN, DeepCNN, MDML). The paper does not describe an independently
retrained post-hoc attacker. On BCI IV 2a (four classes, nine subjects), EEGNet UIA falls from
91.50% to 0.04% and task BCA rises from 61.90% to 68.40%. The headline result is identity removal "to 0.43%"
on four datasets [2]. Closed-set chance for nine subjects is 11.1% (our calculation; the paper does
not state a chance level). Reported values are therefore far below chance. A value well below chance
shows that the measured identifier is systematically wrong; it does not show that identity has been
removed.

**NeuroPriv (Gomasta et al., 2026).** The original representation consists of 17 compact EEG features per window
from EEGMAT (36 subjects, mental arithmetic versus baseline). On it, an independent attacker
reaches subject-identity BA 0.692 (chance 1/36 = 0.028). NeuroPriv expands the features with pairwise
products, encodes them, and suppresses gender, age and identity through gradient-reversal privacy
heads. Leakage is measured by SVM, RF and GB attackers trained independently on the released
representation; the strongest attacker is reported. Identity BA falls to 0.206 while task BA
changes from 0.788 to 0.781. The authors state that identity is "substantially reduced but not
completely removed". They also list single-dataset, closed-set evaluation and an attacker with labeled
auxiliary data from the same population as limitations [3].

**SAFE (Mehlman & Narayanan, 2026).** An EEGNet-based autoencoder with a semi-randomized latent mask
is trained against a frozen adversarial subject classifier to produce an anonymized dataset for
release. The paper defines three threat models with independently trained EEGNet attackers:
*unmasking* (trained on raw data, tested on anonymized data), *inference* (trained on anonymized
data, tested on raw data), and *open eval* (black-box access to SAFE, with training and testing on
anonymized data). On BCI IV 2a (chance 0.111), identity BA is 0.991 undefended, 0.186 under
unmasking, 0.161 under inference and 0.686 under open eval. The authors attribute the weaker
protection on BCI IV 2a and GigaDB to stronger identity markers. On MMI, TSception and FBCNet
attackers were also tested. SAFE reports that prior EEG work typically evaluates a single attack in
which the attacker trains directly on the protected data. It also states that the raw pretraining
data for the unmasking attacker could not be separated from the SAFE training data. White-box
attackers were not evaluated [4].

**Subject re-identification benchmark (Mondair, GitHub).** This unreviewed repository report covers
PhysioNet EEG-MMIDB (104 subjects), BCI IV 2a (9 subjects, two sessions) and Lee 2019 OpenBMI (54
subjects, two sessions). It tests five attacks, three victim model families and four defense
families, including domain-adversarial subject-invariant training, DP-SGD and DP-FedAvg. Attacker
capability is graded from black-box API access through penultimate-embedding access to white-box
access. Cross-session re-identification on IV-2a reaches top-1 0.913 (chance 1/9). Domain-adversarial training
reduces leakage under generic probes but "fail[s] under encoder fine-tuning". The report reproduces
a "−20 pp re-ID at zero task cost" result under generic probes, which collapses to top-1 0.762
under a 15-epoch encoder fine-tune. DP-SGD at ε ≤ 1 blocks the fine-tune attacker at about 6 pp
task-accuracy cost [5].

**Riemannian suppression of cross-session leakage (Muhaimin & Nagarajan, 2026).** This
training-free covariance transform equalizes each subject's Riemannian mean and tangent-space
scatter. It is fitted on session 1 and applied unchanged to session 2, so attackers compare defended
data against defended data. On BCI IV 2a, the worst of five covariance-space attackers falls from
90.8% to 15.6% (chance 11.1%) at a 7.0-point decoding cost. On OpenBMI it falls from 90.8% to 9.9%
(chance 1.85%) at a 2.1-point cost. The authors' reproduction of Meng et al.'s perturbation leaves
every covariance-space attacker at its undefended value (worst case 90.8%) [6]. Its alignment
baselines are summarized in Section 5.

**Adversarial subject-invariant EEG learning (Özdenizci et al., 2020).** Here adversarial inference
serves cross-subject transfer, not privacy. An encoder feeds a task classifier and a subject
adversary. Without the adversary, adversary accuracy among 40 training subjects is 48.5% (EEGNet),
31.4% (DeepConvNet) and 62.6% (ShallowConvNet) on the Cho et al. MI dataset (48 of 52 subjects
used). Increasing λ suppresses adversary accuracy. Leakage is measured only by the co-trained
adversary [7]. The benchmark in [5] treats this method as a defense and tests it under adaptive
attack (see above).

**Not opened.** *Incidental Identity Leakage in EEG Motor-Imagery Decoders* (SSRN abstract 7459682)
returned an HTTP 403 challenge page for both the abstract and the PDF. No claim in this review relies
on it [8].

### Comparison table

"Co-trained" means the identity classifier is trained jointly with, or as part of, the protection
model. "Independent" means an attacker is trained separately after protection.

| Work (status) | Protection method | Datasets | How leakage was measured (identifier; attacker training data) | Reported identity metric | Chance level | Utility measured |
|---|---|---|---|---|---|---|
| Meng et al. 2023 [1] (peer-reviewed) | Sample-/user-wise identity-unlearnable perturbation of released EEG | 7 datasets, 5 paradigms (incl. BCI IV 2a, P300, ERN, SSVEP, neonatal seizure, TUSZ) | Independent User-Classifier retrained on the protected training session and tested on held-out sessions; perturbation crafted with EEGNet, transferred to other backbones | UIA (accuracy): mean 70.01% → ≤ 21.36% | Not reported (closed set) | Yes (BCA) |
| ID-RemovalNet 2025 [2] (peer-reviewed) | Identity decorrelation + GRL + attention feature selection + task-feature re-fusion; alignment step | BCI IV 2a, BCI IV 2b, P300 (8 subj.), ERN (16 subj.) | Identifier of two FC layers attached to the backbone; independent post-hoc attacker not reported; attacker training data not reported | UIA: e.g. 2a/EEGNet 91.50% → 0.04%; headline 0.43% | Not reported (1/9 = 11.1% for 2a by calculation) | Yes (BCA; improves) |
| NeuroPriv 2026 [3] (workshop paper; read as preprint) | Feature expansion + encoder with gradient-reversal heads for gender, age, identity | EEGMAT (36 subj.) | Independent SVM/RF/GB attackers trained on the released representation (labeled auxiliary data from the same population) | Identity BA: 0.692 → 0.206 (strongest attacker) | 0.028 (reported) | Yes (task BA 0.788 → 0.781) |
| SAFE 2026 [4] (peer-reviewed) | Autoencoder with semi-randomized latent masking, adversarial subject model | BCI IV 2a, GigaDB, MMI, SMNI | Independent EEGNet attackers: trained on raw → tested on protected; trained on protected → tested on raw; trained and tested on protected (open eval); TSception/FBCNet on MMI | Identity BA, 2a: 0.991 → 0.186 / 0.161 / 0.686 | 0.111 for 2a (reported) | Yes (task BA, four train/eval combinations) |
| Mondair (unreviewed repository) [5] | Benchmark of defenses: ad hoc transforms, domain-adversarial training, DP-SGD, DP-FedAvg | PhysioNet MMIDB, BCI IV 2a, OpenBMI | Independent probes on victim outputs/embeddings, adaptive encoder fine-tuning, open-set verifier on disjoint subjects, membership inference | Top-k accuracy, ROC-AUC; 2a cross-session top-1 0.913; adversarial defense 0.762 under fine-tune | Reported (e.g. 1/9, 1.85%) | Yes (task accuracy) |
| Muhaimin & Nagarajan 2026 [6] (preprint) | Training-free Riemannian mean alignment and tangent-scatter whitening | BCI IV 2a, OpenBMI | Five independent covariance-space attackers, cross-session S1 → S2 on defended data; trial- and session-level aggregation | Identification accuracy, worst attacker: 90.8% → 15.6% (2a), → 9.9% (OpenBMI) | 11.1%, 1.85% (reported) | Yes (decoding cost 7.0 / 2.1 points) |
| Özdenizci et al. 2020 [7] (peer-reviewed; transfer, not privacy) | Adversarial subject-invariant encoder | Cho et al. MI dataset (48 subj.) | Co-trained adversary only, on training-subject data | Adversary accuracy (e.g. 48.5% EEGNet at λ = 0) | Not reported | Yes (cross-subject accuracy) |
| Sadiq & Toorani 2026 [15] (preprint) | Subject-level DP (Gaussian/Laplace) on patient-level clinical EEG features | Clinical EEG cohort (epilepsy classification) | Not measured; re-identification, linkage and membership analyses left to future work | Not reported | Not reported | Yes (statistical and downstream ML utility) |
| Bay et al. 2026 [16] (peer-reviewed, early version; abstract only) | Participant-specific LDP budget selection (TOPSIS / Fuzzy TOPSIS) | University of Bonn EEG | Not reported in the abstract | Not reported | Not reported | Yes (eight classifiers vs. noise-free baseline) |
| SSRN 7459682 [8] (not opened) | Not opened | Not opened | Not opened | Not opened | Not opened | Not opened |

## 2. Adversarial removal: critiques and alternatives

Elazar and Goldberg show that adversarial training can drive the adversary to chance-level
development accuracy while a post-hoc attacker, trained on the frozen encodings, still recovers
demographic attributes "substantially" above chance. They conclude: "do not rely on the adversarial
training to achieve invariant representation to sensitive features" [9]. Barrett et al. qualify
this result. In their experiments, the diagnostic classifier generalized poorly to new in-domain
samples and new domains, so the detected leakage may depend on correlations specific to the
sample [10]. The measurement implication is that the post-hoc attacker must be evaluated on data
genuinely separate from its training data. Roy and Boddeti replace the likelihood-maximizing
adversary with a maximum-entropy objective, which pushes the adversary's posterior toward uniform,
and analyze its stability. They report reduced leakage on UCI, Extended Yale B and CIFAR-10/100 [11].
Song and Shmatikov show that models trained for simple objectives "overlearn" sensitive attributes.
Censoring those attributes left the models either failing their task or still leaking, and the
authors' de-censoring technique extracts information from censored representations [12].

In EEG, Mondair reports the same pattern under an adaptive attacker that fine-tunes the encoder [5].
SAFE and NeuroPriv evaluate with independently trained attackers and report protection. NeuroPriv
reports identity still well above chance [3, 4]. ID-RemovalNet and Özdenizci et al. report only
identifiers trained with the model [2, 7].

## 3. EEG identifiability across sessions

Maiorana and Campisi evaluate EEG biometrics over the long term: 45 subjects, 5–6 sessions spanning
three years, and four elicitation protocols. They characterize channel uniqueness and permanence
and propose countermeasures against aging effects [13]. Across corpora, MI EEG is highly
identifying. PhysioNet closed-set re-identification reaches 100% top-1 with Riemannian features at
35% task accuracy, and Lee 2019 cross-session verification reaches AUC 0.909 [5]. SAFE's undefended
BCI IV 2a attacker reaches BA 0.991 [4], and Muhaimin and Nagarajan's undefended cross-session
attacker reaches 90.8% [6]. As an earlier security result, Martinovic et al. used consumer-grade EEG
devices to reduce the entropy of private information (e.g., PINs, bank cards) by about 15–40%
relative to random guessing. That work concerns inference of private information, not user
identification [14].

## 4. Privacy mechanisms and overviews for EEG/BCI

Sadiq and Toorani apply subject-level Gaussian and Laplace perturbation to clinical EEG features
under three deployment scenarios. They assess utility only and defer re-identification testing to
future work [15]. Bay et al. select participant-specific local-DP budgets for EEG features using
TOPSIS and Fuzzy TOPSIS on the University of Bonn dataset; only the abstract was read [16]. Mondair
reports that DP-SGD at ε ≤ 1 blocks adaptive re-identification. It also notes that its DP-FedAvg
participant-level ε is informal because the RDP accounting assumes Poisson client sampling, which the
run did not use [5]. Xia et al. survey privacy threats and protection strategies in BCIs [17]. No
randomized-response mechanism applied to BCI commands was found among the opened sources.
Neuro-rights and legal overviews are outside this review because no such source was verified.

## 5. Riemannian and Euclidean alignment as non-adversarial identity reducers

Zanini et al. affine-transform each session's or subject's covariance matrices so they are centered
on a reference matrix, which makes sessions and subjects comparable for transfer learning; only the
abstract was read [18]. Rodrigues et al. propose Riemannian Procrustes Analysis, which matches the
statistical distributions of two datasets of covariance matrices; only the HAL record was read [19].
He and Wu align each subject's trials in Euclidean space without labels (Euclidean alignment, EA).
They report better transfer than a Riemannian alignment approach on two MI datasets and one ERP
dataset [20]. None of these papers evaluates identity.

Muhaimin and Nagarajan evaluate EA and a recentering-and-rescaling variant of RPA without the
rotation step (RPA-RS) as privacy baselines [6]. EA reduces every attacker and improves decoding,
yet leaves the worst attacker at 69.0% on BCI IV 2a and 48.4% on OpenBMI. RPA-RS reaches 46.3% on
2a, but XDAWN still reads 54.5% on OpenBMI. The authors conclude that EA provides "partial and
unquantified protection" [6].

## 6. Positioning of this project

This section was first written on 2026-11-10 before the results of D, E and L were available, and
was updated on 2026-11-11 with those results and with Section 7. BEAT numbers are taken from
`docs/report.md`, which names their result files. "Within-day" means attacker training and test data
from the same recording day (2a: runs 1–3 and 4–6 of session E; Lee2019: offline and online phase of
session 2); "cross-day" means training on the calibration day and testing on the deployment day.

**Replication of established findings.**
- Raw MI representations identify users almost perfectly within a day (BEAT: identity BA 1.00 for
  the 253-d tangent vector on 2a; 0.994 for the 210-d vector on 54 Lee2019 users). Across days the
  same attackers reach 0.951 on 2a and 0.520 on Lee2019 (chance 0.111 and 0.0185). High
  identifiability agrees with EEG biometrics [21–23] and with [4], [5], [6]; the drop across days
  agrees with the session dependence addressed in [13] and [23].
- A co-trained identity adversary can look successful while independent attackers recover identity
  (BEAT: head BA 0.026 versus independent attackers 0.95 at λ = 1 on 2a; 0.040 versus 0.514 within
  a day on 54 users, a gap of 17.6–19.9 pp across days). This is the effect of [9], previously
  reported in EEG under adaptive attack by an unreviewed benchmark [5].
- Recentering and alignment reduce, but do not remove, identity [6].
- A defence fitted to released or reference data can fail against an attacker who adapts later, as
  shown for face-recognition poisoning [26]; BEAT's experiment G shows the analogous effect for a
  reference-session-fitted EEG defence.

**Two evaluation practices.** BEAT's pitfalls are two practices that appear in the opened sources;
we do not claim that they are common across the field, because this review is not a systematic
count. (1) *Co-trained-only measurement:* ID-RemovalNet reports identity only through identifiers
trained with the model [2], as does the transfer-learning work of Özdenizci et al. [7]. (2)
*Reference-session-only attackers:* Muhaimin and Nagarajan evaluate attackers trained on defended
session-1 data [6], and Meng et al. retrain the user classifier on the protected training session
[1]; SAFE states that prior EEG work typically evaluates a single attack in which the attacker trains
directly on the protected data [4]. NeuroPriv [3], SAFE [4] and the benchmark [5] already use
independently trained attackers, and [4] and [5] vary the attacker's data access.

**Contributions not found in the reviewed literature.** These are stated as gaps in the opened
sources, not as priority claims.
- Leakage is measured on outputs collected **after deployment**: attackers are retrained on frozen
  outputs of the deployment session (the main, within-day scenario) and, separately, across days.
  This differs from attackers trained on protected training data [1, 4] and from co-trained
  identifiers [2, 7]. The closest design is the cross-session, defended-versus-defended evaluation
  of [6].
- The co-trained head and independent attackers are compared side by side, at every λ, inside a
  pre-registered protocol on BCI IV 2a, and the gap is confirmed on 54 users. On 2a, a published
  EEG identity-removal result relies on a co-trained identifier [2].
- Identity leakage from **decoded discrete commands** is quantified. A 2026 review names decoded
  outputs as one of the data that can be re-associated with individuals [27]; we found no opened
  source that measures it. On 2a, within session E, identity rises with the number of observed
  commands (0.16–0.20 for one command, 0.21–0.48 for 20; chance 0.11). On 54 users the
  pre-registered command claim C3 was not supported; in C3's cross-day setting (P2-RC, τ = 0.5)
  command identification stayed near chance (0.019–0.029). L's cross-day attackers enrolled on
  in-sample streams; with cross-fitted streams (study K, exploratory) command-only attackers, as
  in C3, stay at 0.008–0.027, while attackers that also see the cue reach 0.046. At τ = 0.8 the
  cross-day estimates at 20 commands are 0.065–0.075 (descriptive; chance 0.0185). Session-level pooling
  of covariance trials [6] is related.
- Within one pooled model, a small task-only bottleneck gives a better privacy–utility trade-off
  than the gradient-reversal, maximum-entropy and ensemble variants tested (2a, exploratory).

**Methods for command-stream defences (evaluated in BEAT's study K, exploratory).**
- *Randomized response* [31] was introduced as a survey technique in which each respondent, by a
  random device with a known probability, is shown one of two complementary statements (membership
  or non-membership of the sensitive group) and answers truthfully whether it applies, without
  revealing which statement was shown. Its per-command analogue applies directly to discrete
  commands. Plan v3 set a
  per-command version aside (`docs/analysis_plan.md`, "Not pursued"); study K adds it as a baseline
  with ε-LDP per command.
- *Equalized-odds post-processing* [28] and its multiclass extension [29] adjust a fixed
  classifier's outputs so that their distribution given the true label does not depend on a group
  attribute. BEAT's group-equalized decoding (EQ) applies this post-processing with the user's
  identity as the attribute: per-user stochastic maps from command to output, chosen by a linear
  program, give every user of a group the same P(output | target) for the calibration-day
  estimates.
- *Perfect privacy.* Rassouli and Gündüz characterise releases that reveal no information about a
  private variable [30]. Within a group, EQ's release meets this condition for identity under the
  fitted distributions. With a single group of all 54 Lee2019 users, the best such release in K was
  "always hold", with no utility (`results/exp_k/frontier.csv`).

BEAT's adaptations are:
- groups of g users with group membership revealed;
- a utility that weights wrong moves by the hold rule's cost;
- cross-fitted calibration streams;
- evaluation by attackers who enrol on the calibration day or on the deployment day.

In the opened sources we found no application of these methods to BCI command streams. K's results
are exploratory, and in the sealed study M, EQ and RR are pre-registered estimates without a
verdict (`docs/report.md` §6.8, §6.10).

**Claims that must be softened.**
- *Generality.* The data are nine subjects in one closed-set dataset and 54 subjects in a second
  (Lee2019, confirmatory replication of four claims, three supported). An attacker's failure does
  not establish anonymization (stated in `README.md`).
- *"Adversarial training fails."* This is supported for gradient reversal on Riemannian features
  (experiment B, confirmatory). Experiments D (MaxEnt [11], ensemble refresh) and E (EEGNet with a
  two-layer identifier) are exploratory: no D variant passed the protection rule in both families,
  and some E settings lowered identity by 7–8 pp relative to λ = 0. The report should claim neither
  that the failure generalizes to other adversarial objectives nor that ID-RemovalNet [2] leaks.
  Experiment E tests the evaluation metric on a common backbone; it does not reproduce
  ID-RemovalNet.
- *"Gradient reversal backfired."* This holds at high λ. The cross-validated operating point
  (λ = 0.01) showed no difference from the λ = 0 bottleneck (identity BA 0.626 vs 0.619).
- *Recentering.* Recentering the full tangent vector left identity BA at 0.998. The reduction to
  0.43 comes from combining recentering with the 8-d bottleneck. Recentering alone should not be
  described as an effective protection. This is consistent with EA leaving 69.0% in [6].
- *Strength of protection.* The best BEAT configuration (the 1-d task-only bottleneck, BA 0.21–0.25
  on 2a, chance 0.11) remains about twice chance. A training-free Riemannian transform reports 15.6%
  on BCI IV 2a against reference-session attackers [6], and DP-SGD at ε ≤ 1 blocks adaptive
  attackers in [5]. BEAT re-implemented [6] only to test its threat model (experiment G) and did not
  evaluate DP-SGD, so BEAT cannot claim the strongest defense. Metrics and scenarios also differ
  (balanced accuracy versus identification accuracy; within-session runs versus cross-session).
- *Cross-day claims.* The main attacker scenario is within-day: runs of the same session E on 2a,
  the offline and online phase of session 2 on Lee2019. Cross-day evidence comes from the auxiliary
  scenario, which the plan says cannot establish protection alone; on Lee2019 the full tangent vector
  gives 0.994 within a day and 0.520 across days. Given [10], leakage statements should name the
  scenario they come from.
- *Command leakage* is above chance but modest at short horizons, and it depends on model and
  confidence threshold τ (22 of 24 settings exceed 2 × chance within five commands, within session E
  on 2a). It was not confirmed on 54 users (claim C3).

## 7. Additional prior work (added 2026-11-11; abstract or record page only)

**EEG biometrics.** Marcel and Millán authenticate persons from EEG with Gaussian mixture models and
maximum a posteriori adaptation, using a single training session, and report that some mental tasks
suit authentication better than others [21]. Bak and Jeong study biometric user identification from
motor-imagery EEG signals and compare feature-extraction and classification methods [22]. Özdenizci
et al. use adversarial inference to learn session-invariant, person-discriminative representations,
because session-specific variability limits EEG identification across longitudinally recorded
sessions [23]. These works treat identity as the target of the system; BEAT treats it as leakage
from a task system.

**Identity protection.** Meng et al. design perturbations that conceal user identity, gender and BCI
experience at once while keeping primary-task accuracy, extending the identity-protection setting
of [1] [24].

**Fixed protection against later attackers.** Radiya-Dixit et al. show that poisoning-based
protection of published face images gives a false sense of security: the images are perturbed once,
before release, and must then fool every future model, including models trained adaptively against
the perturbation or built with later techniques [26]. Pitfall 2 of BEAT (a frozen,
reference-session-fitted EEG defence defeated by a few labelled deployment-period outputs) is an
instance of the same asymmetry in a different threat model.

**BCI privacy review.** Sun et al. (arXiv preprint, May 2026) define protection objects along the BCI
lifecycle, state that neural data, derived representations, model assets and decoded outputs can
be re-associated with individuals, and grade existing protections into four levels of protection
strength [27]. BEAT measures one of these pathways (decoded commands) empirically.

**Output post-processing and perfect privacy.** Hardt et al. define equalized odds and equal
opportunity and show how to optimally adjust any learned predictor using only the joint statistics
of the predictor, the target and the protected attribute [28]. Putzel and Lee extend this
post-processing to multiclass black-box classifiers (SafeAI workshop at AAAI 2022) and report minor
accuracy drops when the dataset is large relative to the number of classes and protected groups
[29]. Rassouli and Gündüz ask how much information about useful data Y can be released while the
release is statistically independent of private data X ("perfect privacy"), with a linear-algebraic
analysis for finite alphabets [30]. Warner introduced randomized response as a survey technique for
eliminating evasive answer bias [31]. In BEAT's setting the protected attribute is the user's
identity and the released output is the decoded command. Section 6 states how study K adapts
these methods (exploratory).

**Data and protocol of study M.** Yang et al. describe the multi-day WBCIC-SHU motor-imagery
dataset (51 two-class participants, three recording days) used in BEAT's sealed replication M [37].
The seal-4 timestamp follows the RFC 3161 time-stamp protocol [38].

**Datasets and methods cited by the report.** BCI Competition IV and its data set 2a (BNCI2014-001)
are described in [25]; the Lee2019/OpenBMI dataset in [32]. The report's methods follow Riemannian
tangent-space classification [33], common spatial patterns [34], gradient reversal [35] and EEGNet
[36].

## References

1. L. Meng, X. Jiang, J. Huang, W. Li, H. Luo, D. Wu, "User Identity Protection in EEG-based
   Brain-Computer Interfaces," *IEEE Trans. Neural Syst. Rehabil. Eng.*, vol. 31, pp. 3576–3586,
   2023. https://doi.org/10.1109/TNSRE.2023.3310883 — full text read at
   https://arxiv.org/abs/2412.09854 (arXiv version).
2. H. Wang, J. Ruan, C. Fan, Y. Cheng, Z. Lv, "ID-RemovalNet: Identity Removal Network for EEG
   Privacy Protection with Enhancing Decoding Tasks," *Proc. IJCAI-25*, pp. 4209–4217, 2025.
   https://doi.org/10.24963/ijcai.2025/469 — PDF: https://www.ijcai.org/proceedings/2025/0469.pdf
3. S. S. Gomasta, B. Chhaglani, P. Shenoy, "NeuroPriv: Adversarial Representation Learning for
   Privacy in Wearable EEG Systems," *Proc. 1st Workshop on Hot Topics in Mobile and Wireless
   Systems for Health*, ACM, pp. 9–14, 2026. **Workshop paper** (Crossref record of
   https://doi.org/10.1145/3842657.3842784, checked 2026-11-11; the full text was read as the
   preprint arXiv:2609.00390, https://arxiv.org/abs/2609.00390).
4. N. Mehlman, S. Narayanan, "Task-Preserving EEG Anonymization Using Latent Feature Masking,"
   *IEEE J. Biomed. Health Inform.*, 2026, early access, pp. 1–10. https://doi.org/10.1109/JBHI.2026.3704845
   — PubMed: https://pubmed.ncbi.nlm.nih.gov/42308059/ ; PDF via
   https://sail.usc.edu/publications/html/b2hd-Mehlman-IEEEJBHI0226.html
5. M. S. Mondair, "Subject Re-Identification Leakage in Motor-Imagery BCI Models: A Multi-Corpus,
   Audit-Clean Benchmark of Attacks and Defenses," GitHub repository with technical report.
   **Unreviewed repository.** https://github.com/manrajmondair/bci-identity-leakage
6. S. S. Muhaimin, A. Nagarajan, "Riemannian Suppression of Cross-Session Identity Leakage in
   Motor-Imagery EEG," arXiv:2610.15626, 2026. **Preprint.** https://arxiv.org/abs/2610.15626
7. O. Özdenizci, Y. Wang, T. Koike-Akino, D. Erdoğmuş, "Learning Invariant Representations from EEG
   via Adversarial Inference," *IEEE Access*, vol. 8, pp. 27074–27085, 2020.
   https://doi.org/10.1109/ACCESS.2020.2971600 — full text read as MERL TR2020-049:
   https://www.merl.com/publications/docs/TR2020-049.pdf
8. "Incidental Identity Leakage in EEG Motor-Imagery Decoders," SSRN abstract 7459682.
   **Not opened** (HTTP 403); authorship and content unverified. https://ssrn.com/abstract=7459682
9. Y. Elazar, Y. Goldberg, "Adversarial Removal of Demographic Attributes from Text Data," *Proc.
   EMNLP*, 2018. https://doi.org/10.18653/v1/D18-1002 — https://aclanthology.org/D18-1002
10. M. Barrett, Y. Kementchedjhieva, Y. Elazar, D. Elliott, A. Søgaard, "Adversarial Removal of
    Demographic Attributes Revisited," *Proc. EMNLP-IJCNLP*, pp. 6329–6334, 2019.
    https://doi.org/10.18653/v1/D19-1662 — https://aclanthology.org/D19-1662
11. P. C. Roy, V. N. Boddeti, "Mitigating Information Leakage in Image Representations: A Maximum
    Entropy Approach," *Proc. CVPR*, pp. 2581–2589, 2019. https://doi.org/10.1109/CVPR.2019.00269 —
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Roy_Mitigating_Information_Leakage_in_Image_Representations_A_Maximum_Entropy_Approach_CVPR_2019_paper.pdf
12. C. Song, V. Shmatikov, "Overlearning Reveals Sensitive Attributes," *Proc. ICLR*, 2020.
    https://openreview.net/forum?id=SJeNz04tDS — https://www.cs.cornell.edu/~shmat/shmat_iclr20.pdf
13. E. Maiorana, P. Campisi, "Longitudinal Evaluation of EEG-based Biometric Recognition," *IEEE
    Trans. Inf. Forensics Security*, vol. 13, no. 5, pp. 1123–1138, 2018.
    https://doi.org/10.1109/TIFS.2017.2778010
14. I. Martinovic, D. Davies, M. Frank, D. Perito, T. Ros, D. Song, "On the Feasibility of
    Side-Channel Attacks with Brain-Computer Interfaces," *Proc. 21st USENIX Security Symp.*,
    pp. 143–158, 2012.
    https://www.usenix.org/conference/usenixsecurity12/technical-sessions/presentation/martinovic
15. N. Sadiq, M. Toorani, "Differentially Private EEG Feature Anonymization: A Privacy–Utility Case
    Study in Clinical Neurophysiology," arXiv:2609.11777, 2026. **Preprint.**
    https://arxiv.org/abs/2609.11777
16. A. Bay, S. Şengül Ayan, S. Demir, "A sensitivity-aware privacy budget selection framework for
    EEG data under local differential privacy," *EURASIP J. Inf. Security*, 2026 (published online
    5 September 2026, no volume assigned in the Crossref record; abstract only read).
    https://doi.org/10.1186/s13635-026-00242-w
17. K. Xia, W. Duch, Y. Sun, K. Xu, W. Fang, H. Luo, Y. Zhang, D. Sang, X. Xu, F. Wang, D. Wu,
    "Privacy-Preserving Brain-Computer Interfaces: A Systematic Review," *IEEE Trans. Comput. Soc.
    Syst.*, vol. 10, no. 5, pp. 2312–2324, 2023. https://doi.org/10.1109/TCSS.2022.3184818 — arXiv
    record (abstract read): https://arxiv.org/abs/2412.11394
18. P. Zanini, M. Congedo, C. Jutten, S. Said, Y. Berthoumieu, "Transfer learning: a Riemannian
    geometry framework with applications to Brain-Computer Interfaces," *IEEE Trans. Biomed. Eng.*,
    vol. 65, no. 5, pp. 1107–1116, 2018. https://doi.org/10.1109/TBME.2017.2742541 — HAL record
    (abstract read): https://hal.science/hal-01923278
19. P. L. C. Rodrigues, C. Jutten, M. Congedo, "Riemannian Procrustes Analysis: Transfer Learning for
    Brain-Computer Interfaces," *IEEE Trans. Biomed. Eng.*, vol. 66, no. 8, pp. 2390–2401, 2019
    (print issue August 2019 per Crossref). https://doi.org/10.1109/TBME.2018.2889705 — HAL record
    (abstract read; lists the year as 2018): https://hal.science/hal-01971856
20. H. He, D. Wu, "Transfer Learning for Brain-Computer Interfaces: A Euclidean Space Data Alignment
    Approach," *IEEE Trans. Biomed. Eng.*, vol. 67, no. 2, pp. 399–410, 2020.
    https://doi.org/10.1109/TBME.2019.2913914 — full text read as arXiv:1808.05464v2
    (2 April 2019): https://arxiv.org/abs/1808.05464
21. S. Marcel, J. del R. Millán, "Person Authentication Using Brainwaves (EEG) and Maximum A
    Posteriori Model Adaptation," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 29, no. 4,
    pp. 743–752, 2007. https://doi.org/10.1109/TPAMI.2007.1012 — PubMed abstract read:
    https://pubmed.ncbi.nlm.nih.gov/17299229/
22. S. Bak, J. Jeong, "User Biometric Identification Methodology via EEG-Based Motor Imagery
    Signals," *IEEE Access*, vol. 11, pp. 41303–41314, 2023.
    https://doi.org/10.1109/ACCESS.2023.3268551 (record page only)
23. O. Özdenizci, Y. Wang, T. Koike-Akino, D. Erdoğmuş, "Adversarial Deep Learning in EEG
    Biometrics," *IEEE Signal Process. Lett.*, vol. 26, no. 5, pp. 710–714, 2019.
    https://doi.org/10.1109/LSP.2019.2906826 — PubMed abstract read:
    https://pubmed.ncbi.nlm.nih.gov/31814690/
24. L. Meng, X. Jiang, T. Jia, D. Wu, "Protecting Multiple Types of Privacy Simultaneously in
    EEG-Based Brain-Computer Interfaces," *Proc. IEEE Int. Conf. Systems, Man, and Cybernetics
    (SMC)*, pp. 2894–2899, 2024. https://doi.org/10.1109/SMC54092.2024.10832088 — abstract read
    at https://arxiv.org/abs/2411.19498
25. M. Tangermann, K.-R. Müller, A. Aertsen, N. Birbaumer, C. Braun, C. Brunner, R. Leeb,
    C. Mehring, K. J. Miller, G. R. Müller-Putz, G. Nolte, G. Pfurtscheller, H. Preissl, G. Schalk,
    A. Schlögl, C. Vidaurre, S. Waldert, B. Blankertz, "Review of the BCI Competition IV,"
    *Front. Neurosci.*, vol. 6, art. 55, 2012. https://doi.org/10.3389/fnins.2012.00055 (record
    page only; describes the competition data sets, including data set 2a = BNCI2014-001)
26. E. Radiya-Dixit, S. Hong, N. Carlini, F. Tramèr, "Data Poisoning Won't Save You From Facial
    Recognition," *Proc. ICLR*, 2022. https://openreview.net/forum?id=B5XahNLmna — abstract read
    at https://arxiv.org/abs/2106.14851
27. L. Sun, X. Mao, S. Zhang, Q. Zeng, M. Zhao, J. Li, W. Dong, "Revisiting Privacy Preservation in
    Brain-Computer Interfaces: Conceptual Boundaries, Risk Pathways, and a Protection-Strength
    Grading Framework," arXiv:2605.11386, 2026. **Preprint** (abstract read).
    https://doi.org/10.48550/arXiv.2605.11386
28. M. Hardt, E. Price, N. Srebro, "Equality of Opportunity in Supervised Learning," *Advances in
    Neural Information Processing Systems 29 (NIPS 2016)*, 2016.
    https://proceedings.neurips.cc/paper_files/paper/2016/hash/6a9659feb1216f14f7384ba499518b38-Abstract.html
    — abstract read at https://arxiv.org/abs/1610.02413
29. P. Putzel, S. Lee, "Blackbox Post-Processing for Multiclass Fairness," *Proc. Workshop on
    Artificial Intelligence Safety 2022 (SafeAI 2022), co-located with AAAI-22*, CEUR Workshop
    Proceedings vol. 3087, 2022. **Workshop paper.** https://ceur-ws.org/Vol-3087/paper_36.pdf —
    abstract read at https://arxiv.org/abs/2201.04461
30. B. Rassouli, D. Gündüz, "On Perfect Privacy," *IEEE J. Sel. Areas Inf. Theory*, vol. 2, no. 1,
    pp. 177–191, 2021. https://doi.org/10.1109/JSAIT.2021.3053432 — abstract read at
    https://arxiv.org/abs/1712.08500
31. S. L. Warner, "Randomized Response: A Survey Technique for Eliminating Evasive Answer Bias,"
    *J. Amer. Statist. Assoc.*, vol. 60, no. 309, pp. 63–69, 1965.
    https://doi.org/10.1080/01621459.1965.10480775 (record page only)
32. M.-H. Lee, O.-Y. Kwon, Y.-J. Kim, H.-K. Kim, Y.-E. Lee, J. Williamson, S. Fazli, S.-W. Lee,
    "EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI
    illiteracy," *GigaScience*, vol. 8, no. 5, giz002, 2019.
    https://doi.org/10.1093/gigascience/giz002 — data: https://doi.org/10.5524/100542
33. A. Barachant, S. Bonnet, M. Congedo, C. Jutten, "Multiclass Brain–Computer Interface
    Classification by Riemannian Geometry," *IEEE Trans. Biomed. Eng.*, vol. 59, no. 4,
    pp. 920–928, 2012. https://doi.org/10.1109/TBME.2011.2172210
34. H. Ramoser, J. Müller-Gerking, G. Pfurtscheller, "Optimal Spatial Filtering of Single Trial EEG
    During Imagined Hand Movement," *IEEE Trans. Rehabil. Eng.*, vol. 8, no. 4, pp. 441–446, 2000.
    https://doi.org/10.1109/86.895946
35. Y. Ganin, V. Lempitsky, "Unsupervised Domain Adaptation by Backpropagation," *Proc. ICML*,
    PMLR vol. 37, pp. 1180–1189, 2015. https://proceedings.mlr.press/v37/ganin15.html
36. V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, B. J. Lance, "EEGNet: a
    compact convolutional neural network for EEG-based brain–computer interfaces," *J. Neural
    Eng.*, vol. 15, no. 5, 056013, 2018. https://doi.org/10.1088/1741-2552/aace8c
37. B. Yang, F. Rong, Y. Xie, D. Li, J. Zhang, F. Li, G. Shi, X. Gao, "A multi-day and high-quality
    EEG dataset for motor imagery brain-computer interface," *Sci. Data*, vol. 12, 488, 2025.
    https://doi.org/10.1038/s41597-025-04826-y. *Peer-reviewed* (data descriptor). Data: B. Yang,
    F. Rong, "WBCIC-SHU Motor Imagery Dataset," version 5, figshare+, 2024, CC BY 4.0.
    https://doi.org/10.25452/figshare.plus.22671172.v5
38. C. Adams, P. Cain, D. Pinkas, R. Zuccherato, "Internet X.509 Public Key Infrastructure
    Time-Stamp Protocol (TSP)," RFC 3161, RFC Editor, Aug. 2001. https://doi.org/10.17487/RFC3161

## Appendix: reference resolution check

Output of `python scripts/check_references.py`, run on 2026-11-11 06:04 KST (21:04 UTC on
2026-11-10; the timestamp in the first line is UTC) and recorded unchanged in
`results/reference_check/references_20261110T2104Z.txt`. The script runs the reference check of
`beat.verify` (also available as `python -m beat.verify --references`). It resolves every DOI through the doi.org handle API (publishers often block
scripted requests, the handle API does not), every arXiv identifier through the arXiv API, and every
other URL in the reference lists of `README.md`, `docs/report.md` and this file by an HTTP GET
(status below 400; PubMed answers 203). Bibliographic fields (authors, title, venue, volume, issue,
pages, year) were compared by hand with the Crossref record of each DOI and the arXiv record of
each preprint on the same day, and the reference lists were corrected to those records.

```text
reference check, 2026-11-10 21:04 UTC
  ok   arxiv 1610.02413 -> Equality of Opportunity in Supervised Learning
  ok   arxiv 1712.08500 -> On Perfect Privacy
  ok   arxiv 1808.05464 -> Transfer Learning for Brain-Computer Interfaces: A Euclidean Space Data Alignment Approach
  ok   arxiv 2106.14851 -> Data Poisoning Won't Save You From Facial Recognition
  ok   arxiv 2201.04461 -> Blackbox Post-Processing for Multiclass Fairness
  ok   arxiv 2411.19498 -> Protecting Multiple Types of Privacy Simultaneously in EEG-based Brain-Computer Interfaces
  ok   arxiv 2412.09854 -> User Identity Protection in EEG-based Brain-Computer Interfaces
  ok   arxiv 2412.11394 -> Privacy-Preserving Brain-Computer Interfaces: A Systematic Review
  ok   arxiv 2605.11386 -> Revisiting Privacy Preservation in Brain-Computer Interfaces: Conceptual Boundaries, Risk Pathways, and a Protection-Strength Grading Framework
  ok   arxiv 2609.00390 -> NeuroPriv: Adversarial Representation Learning for Privacy in Wearable EEG Systems
  ok   arxiv 2609.11777 -> Differentially Private EEG Feature Anonymization: A Privacy-Utility Case Study in Clinical Neurophysiology
  ok   arxiv 2610.15626 -> Riemannian Suppression of Cross-Session Identity Leakage in Motor-Imagery EEG
  ok   doi   10.1038/s41597-025-04826-y -> https://www.nature.com/articles/s41597-025-04826-y
  ok   doi   10.1080/01621459.1965.10480775 -> http://www.tandfonline.com/doi/abs/10.1080/01621459.1965.10480775
  ok   doi   10.1088/1741-2552/aace8c -> https://iopscience.iop.org/article/10.1088/1741-2552/aace8c
  ok   doi   10.1093/gigascience/giz002 -> https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giz002/5304369
  ok   doi   10.1109/86.895946 -> https://ieeexplore.ieee.org/document/895946/
  ok   doi   10.1109/ACCESS.2020.2971600 -> https://ieeexplore.ieee.org/document/8981912/
  ok   doi   10.1109/ACCESS.2023.3268551 -> https://ieeexplore.ieee.org/document/10105200/
  ok   doi   10.1109/CVPR.2019.00269 -> https://ieeexplore.ieee.org/document/8953999/
  ok   doi   10.1109/JBHI.2026.3704845 -> https://ieeexplore.ieee.org/document/11569012/
  ok   doi   10.1109/JSAIT.2021.3053432 -> https://ieeexplore.ieee.org/document/9336011/
  ok   doi   10.1109/LSP.2019.2906826 -> https://ieeexplore.ieee.org/document/8675355/
  ok   doi   10.1109/SMC54092.2024.10832088 -> https://ieeexplore.ieee.org/document/10832088/
  ok   doi   10.1109/TBME.2011.2172210 -> http://ieeexplore.ieee.org/document/6046114/
  ok   doi   10.1109/TBME.2017.2742541 -> https://ieeexplore.ieee.org/document/8013808/
  ok   doi   10.1109/TBME.2018.2889705 -> https://ieeexplore.ieee.org/document/8588384/
  ok   doi   10.1109/TBME.2019.2913914 -> https://ieeexplore.ieee.org/document/8701679/
  ok   doi   10.1109/TCSS.2022.3184818 -> https://ieeexplore.ieee.org/document/9808103/
  ok   doi   10.1109/TIFS.2017.2778010 -> http://ieeexplore.ieee.org/document/8120145/
  ok   doi   10.1109/TNSRE.2023.3310883 -> https://ieeexplore.ieee.org/document/10236508/
  ok   doi   10.1109/TPAMI.2007.1012 -> http://ieeexplore.ieee.org/document/4107576/
  ok   doi   10.1145/3842657.3842784 -> https://dl.acm.org/doi/10.1145/3842657.3842784
  ok   doi   10.1186/s13635-026-00242-w -> https://link.springer.com/10.1186/s13635-026-00242-w
  ok   doi   10.17487/RFC3161 -> https://www.rfc-editor.org/info/rfc3161
  ok   doi   10.18653/v1/D18-1002 -> http://aclweb.org/anthology/D18-1002
  ok   doi   10.18653/v1/D19-1662 -> https://www.aclweb.org/anthology/D19-1662
  ok   doi   10.24963/ijcai.2025/469 -> https://www.ijcai.org/proceedings/2025/469
  ok   doi   10.25452/figshare.plus.22671172.v5 -> https://plus.figshare.com/articles/dataset/Brain_Computer_Interface_Motor_Imagery-EEG_Dataset/22671172/5
  ok   doi   10.3389/fnins.2012.00055 -> http://journal.frontiersin.org/article/10.3389/fnins.2012.00055/abstract
  ok   doi   10.48550/arXiv.2605.11386 -> https://arxiv.org/abs/2605.11386
  ok   doi   10.5524/100542 -> http://gigadb.org/dataset/100542
  ok   url   https://aclanthology.org/D18-1002 -> HTTP 200
  ok   url   https://aclanthology.org/D19-1662 -> HTTP 200
  ok   url   https://ceur-ws.org/Vol-3087/paper_36.pdf -> HTTP 200
  ok   url   https://github.com/manrajmondair/bci-identity-leakage -> HTTP 200
  ok   url   https://hal.science/hal-01923278 -> HTTP 200
  ok   url   https://hal.science/hal-01971856 -> HTTP 200
  ok   url   https://openaccess.thecvf.com/content_CVPR_2019/papers/Roy_Mitigating_Information_Leakage_in_Image_Representations_A_Maximum_Entropy_Approach_CVPR_2019_paper.pdf -> HTTP 200
  ok   url   https://openreview.net/forum?id=B5XahNLmna -> HTTP 200
  ok   url   https://openreview.net/forum?id=SJeNz04tDS -> HTTP 200
  ok   url   https://proceedings.mlr.press/v37/ganin15.html -> HTTP 200
  ok   url   https://proceedings.neurips.cc/paper_files/paper/2016/hash/6a9659feb1216f14f7384ba499518b38-Abstract.html -> HTTP 200
  ok   url   https://pubmed.ncbi.nlm.nih.gov/17299229/ -> HTTP 203
  ok   url   https://pubmed.ncbi.nlm.nih.gov/31814690/ -> HTTP 203
  ok   url   https://pubmed.ncbi.nlm.nih.gov/42308059/ -> HTTP 203
  ok   url   https://sail.usc.edu/publications/html/b2hd-Mehlman-IEEEJBHI0226.html -> HTTP 200
  skip url   https://ssrn.com/abstract=7459682 -> listed as not opened; not cited
  ok   url   https://www.cs.cornell.edu/~shmat/shmat_iclr20.pdf -> HTTP 200
  ok   url   https://www.ijcai.org/proceedings/2025/0469.pdf -> HTTP 200
  ok   url   https://www.merl.com/publications/docs/TR2020-049.pdf -> HTTP 200
  ok   url   https://www.usenix.org/conference/usenixsecurity12/technical-sessions/presentation/martinovic -> HTTP 200
references: pass
```
