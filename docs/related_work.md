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

## 8. Methods imported by the audit tool (added 2026-11-12)

BEAT v3 adds `beat_audit`, a dataset-agnostic tool that measures open-set linkage: an attacker who
never enrolled the target users decides whether two windows of transmitted outputs come from the
same person (`docs/plan_v3.md` §2 lists the imported methods, §3 specifies the tool;
`beat_audit/README.md` is the builder-facing description). This section states, for each imported
method, what the source defines, how the tool uses it, what BEAT adapts and where the analogy
stops. Standards, peer-reviewed papers, evaluation plans and unreviewed sources are marked in the
reference list, together with how much of each source was read. No result of the audit runs is
quoted here.

**8.1 Biometric performance testing.** ISO/IEC 19795-1:2021 sets out principles and framework for
testing and reporting biometric performance [39]. It defines the detection error trade-off (DET) as
the relationship between false-negative and false-positive errors of a binary classification
system as the discrimination threshold varies (clause 3.28); its table of contents lists the
formation of mated and non-mated comparison trials (clause 8.4) and the false non-match and false
match rates as verification metrics (clauses 9.4.1 and 9.4.2). The equal error rate is not among the defined terms of
the parts we read; the tool reports it as the conventional point of the DET where both error rates
are equal. *Use* (plan §2, §3 "Protocol", "Metrics and inference"): a reference window of user u
against a probe window is a mated comparison when the probe is from u and a non-mated comparison
when it is from another target user; the tool reports FMR, FNMR, EER, TPR at fixed FPR and DET
curves. *Adaptation*: the "biometric system" is an attacker the tool builds from transmitted
outputs and, in BEAT's audits, each trial's task cue (an attacker without the cues was not
measured); comparators are fitted on development users only, so every scored pair involves
users the comparator never saw, and the deciding statistic is a per-user AUC with a matching
permutation test rather than a pooled error rate. *Limit*: 19795-1 evaluates a system built to
recognise its enrolled users; BEAT's outputs were produced by a task decoder that was never meant
to carry identity, and a low error rate here is a finding against the BCI, not a performance claim.

**8.2 Unlinkability of protected templates.** ISO/IEC 24745:2022 defines unlinkability as the
property of two or more biometric references that they cannot be linked to each other or to the
subject(s) from whom they were derived (clause 3.38) [40]. Gomez-Barrero et al. turn this into two
measures computed from the mated and non-mated score distributions of a linkage function: a local,
score-wise measure D↔(s) based on the likelihood ratio between the two distributions, and a global
measure D↔sys, independent of the score domain, both in a closed range [41]. *Use* (plan §2; §3,
"D_sys"): D(s) = 2ωLR(s) / (1 + ωLR(s)) − 1 where ωLR(s) > 1, else 0, with ω = 1, integrated
against the mated score density; LR(s) comes from weighted histograms on fixed bins. *Adaptation*:
the "references" are windows of a transmitted command or feature stream, and the linkage function
is the tool's own comparator. D_sys is therefore a linkability index of a stream under a stated
attacker, not a property of a template-protection scheme. *Limit*: the source assumes a protection
scheme that is designed to be unlinkable and asks whether it succeeds; a task BCI makes no such
claim, and D_sys is bounded by the strength of the comparator, so a low value is not evidence of
unlinkability by other attackers. The tool prints it as an index, never as an identification rate.

**8.3 Attacker knowledge in speaker anonymisation.** The VoicePrivacy 2020 evaluation plan assumes
an attacker with one or more anonymised trial utterances and, possibly, enrolment utterances that
may or may not be anonymised [42]. The 2022 plan records that the 2020 evaluation used three attack
models, *ignorant*, *lazy-informed* and *semi-informed*, corresponding to attackers with different
knowledge; it adopts a semi-informed attacker whose verification model is retrained on anonymised
data because this gave a lower EER, "i.e., a stronger attack", and considers it the most reliable
for privacy assessment [43]. The 2024 plan keeps the semi-informed attacker and warns that a badly
trained attacker model can overestimate the EER and give a false sense of privacy [44]. Srivastava
et al. define ignorant, semi-informed and informed attackers by their knowledge of the voice
conversion and note that an ignorant attacker is very weak [45]. *Use* (plan §2): the tool's
attacker grid varies enrolment (none, reference day, deployment day) and the day relation
(same-day against cross-day), and always prints same-day beside cross-day. *Adaptation*: the
grid orders attackers by what they know about the user and the recording day, not by knowledge
of an anonymisation algorithm; it is an analogy by attacker knowledge, not the VoicePrivacy
protocol. BEAT's own development run showed the same effect: on recentred features, the
mean-based feature comparators understated exposure relative to the command attacker on the
commands decoded from the same features. The plan therefore added richer window summaries and the
data-processing guard of 8.9 before the frozen run (plan, State, 2026-11-11). *Limit*: speech
anonymisation transforms a signal that is meant to stay intelligible; BEAT's outputs are a
low-rate stream of decoded commands or features, and no protocol, data split or metric of the
VoicePrivacy challenges is reused.

**8.4 Forensic likelihood ratios.** Brümmer and du Preez's "Application-independent evaluation of
speaker detection" [46] is the journal source of the log-likelihood-ratio cost Cllr; only its
record page was checked, so its content is cited through two related sources. Brümmer's Odyssey
2004 paper of the same title proposes a logarithmic utility (a strictly proper scoring rule) for
evaluating likelihood-ratio outputs of speaker detectors and gives it an information-theoretic
interpretation [47]. Brümmer and du Preez show that pool-adjacent-violators (PAV) calibration is
optimal for every regular binary proper scoring rule, also when calibrating log-likelihood ratios,
where optimality holds for any prior [48]; this is what makes the PAV-calibrated cost a minimum,
Cllr_min. Ramos and Gonzalez-Rodriguez describe calibration as a property of a set of
likelihood-ratio values and measure it with empirical cross-entropy (ECE) plots [49]. *Use* (plan
§3, "Calibration"): a one-dimensional logistic regression fitted on development users gives an
estimated log likelihood ratio per comparison; Cllr is computed on target users, and Cllr_min
replaces the fitted map by weighted PAV on the target scores themselves. *Adaptation*: the
evidence is about whether two windows share a source, not about a suspect, and Cllr_min is an
oracle diagnostic that feeds nothing back into the attacker; Cllr − Cllr_min is read as the
calibration loss when a development-day map is carried to another day. *Limit*: in forensic use
the likelihood ratio is reported to a court for one case; the tool's ratios are estimated, valid
only as far as the logistic form and the development users transfer, and are summaries over
pooled comparisons, not case evidence.

**8.5 Sequential analysis.** The tool's sequential attacker is a sequential probability ratio
test in the sense of Wald [50]. Only the record page of [50] was checked, so the error bounds are
taken from Ville's inequality as stated by Ramdas et al.: for a test martingale M under P and any
α ≥ 1, P(sup_t M_t ≥ α) ≤ 1/α (their eq. 8), and for a point null against a point alternative the
likelihood ratio is such a martingale [51]. *Use* (plan §3, "Commands only, sequential attacker"):
probe commands are added one at a time to a cumulative Dirichlet-categorical log likelihood ratio,
which equals the log Bayes factor of 8.7; the attacker decides "link" at log((1 − β)/α) and "no
link" at log(β/(1 − α)) with α = β = 0.05, truncated at the probe length, and reports realised
FMR and FNMR, the share decided and the number of commands to a decision. *Adaptation*: under the
generative model, Ville's inequality bounds the two error rates by α/(1 − β) and β/(1 − α); the
tool's validation checks these bounds on simulated streams. *Limit*: real command streams drift
across days and their commands are serially dependent, so the generative model does not hold; the
realised error rates are operating characteristics of this attacker on these data, not guarantees.

**8.6 Large-population biometrics.** Grother and Phillips give binomial models of open- and
closed-set identification as functions of database size, match rank and operating threshold, note
verification as a special case of open-set identification, and find that the binomial model
approximates performance at low false match rates but underestimates closed-set identification
rates, implicating its iid assumption [52]. *Use* (plan §2; §3, "Gallery module"): the rank-1 rate
at gallery size N is predicted from the mated and non-mated score distributions under
independence, with ties broken uniformly. *Adaptation*: the prediction is first checked against the
measured 1:N rank-1 rate of the same comparator on held-out galleries of each fold; if any
difference exceeds a fixed tolerance, the report prints "independence model not supported" and no
larger N. *Limit*: the gallery sizes the cohorts allow are small; larger N is printed only as a
sensitivity scenario under the independence assumption, never as a population risk.

**8.7 Bayesian comparison of categorical counts.** Minka derives the Dirichlet-multinomial (Pólya)
distribution as a Dirichlet compounded with a multinomial and gives its marginal likelihood in
closed form with gamma functions [53]; Mosimann's study of the compound multinomial distribution
is the earlier journal source [54]. Ommen and Saunders distinguish the *common-source* problem,
whether two sets of evidence of unknown source share a common but unknown origin in a relevant
background population, from the *specific-source* problem [55]. *Use* (plan §3, "Command
comparator"): the same-source against different-source Bayes factor of a reference count vector x
and a probe count vector y under a shared Dirichlet prior α, BF = B(α + x + y) · B(α) /
(B(α + x) · B(α + y)), with α fitted by maximising the Dirichlet-multinomial marginal likelihood
of the development users' windows. *Adaptation*: open-set linkage is the common-source question
(neither window has a known source); the prior is the background population of development
users, and with cue labels the counts and the prior are kept per cue. *Limit*: the same-source
hypothesis assumes one command distribution in both windows; a user whose command mix drifts
between days violates it, so a drift-aware attacker may find more than this comparator. We did
not find, among the sources we reviewed, a published use of this Bayes factor for linking BCI
command streams.

**8.8 Covariance shrinkage.** Ledoit and Wolf introduce a linear shrinkage estimator of a
covariance matrix that is well-conditioned and, as the numbers of observations and variables grow
together, has asymptotically uniformly minimum quadratic risk within its class [56]. *Use* (plan
§3, State 2026-11-11; `beat_audit/README.md`): the LDA comparators whiten with a within-user
Ledoit-Wolf covariance estimated on development users only, and reports record the estimator, the
shrinkage coefficient and the isotropic target. *Adaptation*: none beyond the choice of a
within-user (rather than total) covariance; it replaced a fixed eigenvalue floor after a
development finding. *Limit*: it is a conditioning device for the comparator, not an identity
measure; its asymptotic optimality says nothing about how strong the resulting attacker is.

**8.9 Data-processing inequality as an audit guard.** The data-processing inequality states that
processing cannot increase information: for a Markov chain X → Y → Z, I(X; Y) ≥ I(X; Z) [57, §2.8].
*Use* (plan §3, "Guards from BEAT's findings"): commands are a function of the features they are
decoded from, so the best feature attacker cannot do worse than the best command attacker. When
the builder declares a command stream as derived from a feature stream and the command attacker
scores higher, `compare` reports that the feature attacker is too weak and that the feature
stream's exposure is at least the command stream's. *Adaptation*: the inequality is applied to
attacker performance rather than to mutual information; the step holds because a feature attacker
who knows the decoder can reproduce the commands and every command attacker. *Limit*: the tool's
comparators are fixed families, not optimal attackers, so the guard detects an attacker that is
too weak; it does not measure the gap, and it relies on the builder's declaration of which stream
is derived from which.

**8.10 Regulation and guidance.** The report's reading connects to five instruments. BEAT draws no
legal conclusion: the tool measures linkage under stated attackers, an input to a re-identification
assessment, not compliance.

- UNESCO's Recommendation on the Ethics of Neurotechnology (adopted 11 November 2025, 43rd session
  of the General Conference) describes neural data as data about the structure, activity and
  function of the nervous system gathered through neurotechnology (definition 5), holds that their
  collection, processing and sharing require prior, free and informed consent (paragraph 49), and
  lists privacy impact assessments that evaluate and mitigate risks to privacy among the
  assessments Member States should promote (paragraph 80(d)) [58].
- Colorado HB24-1058 (signed 17 April 2024) adds "biological data", which includes "neural data",
  to the sensitive data of the Colorado Privacy Act; its legislative declaration states that every human
  brain is unique, so neural data is specific to the individual from whom it was collected [59].
- California SB 1223 adds a consumer's neural data, "information that is generated by measuring
  the activity of a consumer's central or peripheral nervous system, and that is not inferred from
  nonneural information", to sensitive personal information under Cal. Civ. Code §1798.140(ae);
  the same section defines deidentified information as information that cannot reasonably be used
  to infer information about, or otherwise be linked to, a particular consumer [60].
- GDPR Recital 26 requires account of "all the means reasonably likely to be used, such as singling
  out", by the controller or another person, to identify a natural person, judged by objective
  factors such as cost, time and the available technology [61].
- The Korean Personal Information Protection Commission's pseudonymised-information guideline
  (fully revised March 2026, risk-based) asks the provider of pseudonymised information to weigh
  the purpose, the processing environment and the information the recipient holds or can access,
  and to lower the risk of re-identification to a reasonable level; it notes that the same data
  item carries a different re-identification risk depending on its combinability with other
  information [62]. We found no provision specific to neural data in it.

An audit result, such as linkage detected at a given window length under a given attacker, is one
observation about the "means reasonably likely to be used" or the risk of re-identification;
whether a transmitted command stream is personal, sensitive, deidentified or pseudonymised
information under any of these instruments is outside what BEAT measures.

**8.11 Closest EEG identity and permanence benchmarks.** Among the sources we reviewed, the
closest evaluation resources are biometric. M3CV is a multi-subject, multi-session, multi-task EEG
database with an identification and verification competition, built to characterise intra- and
inter-subject variability across sessions [63]. NeuroIDBench benchmarks brainwave authentication
models on nine datasets under known- and unknown-attacker models and across sessions [64]. Fallahi
et al. evaluate brainwave authentication on a large multi-session dataset recorded over five years
and report that the EER increases with the time between enrolment and test [65]. Ugail and
Howard's repository tests cross-session permanence under one fixed verification protocol and
reads performance through Cllr with its Cllr_min / calibration decomposition (unreviewed; the
accompanying paper is under review) [66]. *Difference*: these resources evaluate systems whose
purpose is to recognise people from raw or minimally processed EEG, with an enrolment step and
performance as the goal. `beat_audit` evaluates the transmitted outputs of a task BCI that is not
meant to carry identity: its input is the transmitted stream (with each trial's task cue in BEAT's audits), the attacker is built by the tool
from users it never saw, the builder hopes for a low score, and a non-detection is reported as
insufficient evidence, not absence. The cross-session drift these benchmarks document is the
reason the tool always reports same-day beside cross-day.

**Summary.**

| Method | Source | Role in the tool | Where the analogy stops |
|---|---|---|---|
| Mated / non-mated comparisons, FMR, FNMR, DET (EER as its conventional equal-error point, not a 19795-1 term in the parts read) | ISO/IEC 19795-1 [39] | Open-set linkage of two output windows | The "system" is the tool's attacker, not a recogniser the builder built |
| Unlinkability, D↔(s), D↔sys | ISO/IEC 24745 [40]; Gomez-Barrero et al. [41] | One-number linkability index of a stream | Not a property of a protection scheme; bounded by the comparator |
| Attackers graded by knowledge | VoicePrivacy plans [42]–[44]; Srivastava et al. [45] | Enrolment × day-relation attacker grid | Analogy by knowledge only; no VoicePrivacy protocol or metric reused |
| Cllr, Cllr_min under PAV, calibration | Brümmer and du Preez [46]–[48]; Ramos and Gonzalez-Rodriguez [49] | Evidence per comparison in bits; calibration loss across days | Estimated, pooled ratios; not case evidence |
| SPRT, Ville's inequality | Wald [50]; Ramdas et al. [51] | Commands until a link decision; realised versus nominal error | Bounds hold under the generative model only; drift and dependence break it |
| Rank-1 rate at gallery size N | Grother and Phillips [52] | Checked prediction; larger N as a scenario | Independence assumption; no population risk estimate |
| Dirichlet-multinomial same-source Bayes factor | Minka [53]; Mosimann [54]; Ommen and Saunders [55] | Command comparator: closed-form scoring with a development-fitted prior | Assumes one command distribution per user across windows |
| Linear covariance shrinkage | Ledoit and Wolf [56] | Within-user covariance of the LDA comparators | Conditioning only; says nothing about attacker strength |
| Data-processing inequality | Cover and Thomas [57] | Guard: feature exposure ≥ command exposure | Holds for optimal attackers; detects a weak one, does not size the gap |
| Neural-data and re-identification instruments | [58]–[62] | Report reading of "reasonably likely means" | An input to an assessment; no legal conclusion |
| EEG identity and permanence benchmarks | [63]–[66] | Closest existing evaluations | Biometric systems, not a task BCI's transmitted outputs |

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
39. ISO/IEC 19795-1:2021, *Information technology — Biometric performance testing and reporting —
    Part 1: Principles and framework*, edition 2.0, 2021-05-27. **Standard.** IEC webstore record:
    https://webstore.iec.ch/en/publication/69063 — publisher preview read (table of contents and
    clauses 1–6): https://standards.iteh.ai/catalog/standards/iso/62f63185-3037-442c-8a85-96040c86e5ce/iso-iec-19795-1-2021
40. ISO/IEC 24745:2022, *Information security, cybersecurity and privacy protection — Biometric
    information protection*, edition 2.0, 2022-02-08. **Standard.** IEC webstore record:
    https://webstore.iec.ch/en/publication/74235 — publisher preview read (table of contents and
    clauses 1–5): https://standards.iteh.ai/catalog/standards/iso/fb018742-972d-4e70-82d3-b8fe3ecc2b3b/iso-iec-24745-2022
41. M. Gomez-Barrero, J. Galbally, C. Rathgeb, C. Busch, "General Framework to Evaluate
    Unlinkability in Biometric Template Protection Systems," *IEEE Trans. Inf. Forensics
    Security*, vol. 13, no. 6, pp. 1406–1420, 2018. https://doi.org/10.1109/TIFS.2017.2788000 —
    full text read as arXiv:2311.04633 (author version): https://arxiv.org/abs/2311.04633
42. N. Tomashenko, B. M. L. Srivastava, X. Wang, E. Vincent, A. Nautsch, J. Yamagishi, N. Evans,
    J. Patino, J.-F. Bonastre, P.-G. Noé, M. Todisco, "The VoicePrivacy 2020 Challenge Evaluation
    Plan," arXiv:2205.07123, 2022. **Evaluation plan** (not peer-reviewed; full text read).
    https://arxiv.org/abs/2205.07123
43. N. Tomashenko, X. Wang, X. Miao, H. Nourtel, P. Champion, M. Todisco, E. Vincent, N. Evans,
    J. Yamagishi, J.-F. Bonastre, "The VoicePrivacy 2022 Challenge Evaluation Plan,"
    arXiv:2203.12468, 2022. **Evaluation plan** (not peer-reviewed; full text read).
    https://arxiv.org/abs/2203.12468
44. N. Tomashenko, X. Miao, P. Champion, S. Meyer, X. Wang, E. Vincent, M. Panariello, N. Evans,
    J. Yamagishi, M. Todisco, "The VoicePrivacy 2024 Challenge Evaluation Plan," arXiv:2404.02677,
    2024. **Evaluation plan** (not peer-reviewed; full text read). https://arxiv.org/abs/2404.02677
45. B. M. L. Srivastava, N. Vauquier, M. Sahidullah, A. Bellet, M. Tommasi, E. Vincent,
    "Evaluating Voice Conversion-Based Privacy Protection against Informed Attackers," *Proc. IEEE
    ICASSP*, pp. 2802–2806, 2020. https://doi.org/10.1109/ICASSP40776.2020.9053868 — full text
    read as arXiv:1911.03934: https://arxiv.org/abs/1911.03934
46. N. Brümmer, J. du Preez, "Application-independent evaluation of speaker detection," *Computer
    Speech & Language*, vol. 20, no. 2–3, pp. 230–275, 2006.
    https://doi.org/10.1016/j.csl.2005.08.001 (record page only)
47. N. Brümmer, "Application-independent evaluation of speaker detection," *Proc. Odyssey 2004:
    The Speaker and Language Recognition Workshop*, pp. 33–40, 2004. ISCA Archive (abstract read):
    https://www.isca-archive.org/odyssey_2004/brummer04_odyssey.html
48. N. Brümmer, J. du Preez, "The PAV algorithm optimizes binary proper scoring rules,"
    arXiv:1304.2331, 2013. **Preprint** (full text read). https://arxiv.org/abs/1304.2331
49. D. Ramos, J. Gonzalez-Rodriguez, "Reliable support: Measuring calibration of likelihood
    ratios," *Forensic Sci. Int.*, vol. 230, no. 1–3, pp. 156–169, 2013.
    https://doi.org/10.1016/j.forsciint.2013.04.014 — PubMed abstract read:
    https://pubmed.ncbi.nlm.nih.gov/23664798/
50. A. Wald, "Sequential Tests of Statistical Hypotheses," *Ann. Math. Statist.*, vol. 16, no. 2,
    pp. 117–186, 1945. https://doi.org/10.1214/aoms/1177731118 (record page only)
51. A. Ramdas, P. Grünwald, V. Vovk, G. Shafer, "Game-Theoretic Statistics and Safe Anytime-Valid
    Inference," *Statist. Sci.*, vol. 38, no. 4, pp. 576–601, 2023.
    https://doi.org/10.1214/23-STS894 — full text read as arXiv:2210.01948:
    https://arxiv.org/abs/2210.01948
52. P. Grother, P. J. Phillips, "Models of large population recognition performance," *Proc. IEEE
    CVPR*, vol. 2, pp. 68–75, 2004. https://doi.org/10.1109/CVPR.2004.1315146 (abstract read in
    the OpenAlex record of the DOI)
53. T. P. Minka, "Estimating a Dirichlet distribution," technical report, 2000 (revised 2003, 2009,
    2012). **Technical report** (not peer-reviewed; full text read).
    https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf
54. J. E. Mosimann, "On the compound multinomial distribution, the multivariate β-distribution, and
    correlations among proportions," *Biometrika*, vol. 49, no. 1–2, pp. 65–82, 1962.
    https://doi.org/10.1093/biomet/49.1-2.65 (record page only)
55. D. M. Ommen, C. P. Saunders, "Building a unified statistical framework for the forensic
    identification of source problems," *Law, Probability and Risk*, vol. 17, no. 2, pp. 179–197,
    2018. https://doi.org/10.1093/lpr/mgy008 (abstract read in the OpenAlex record of the DOI)
56. O. Ledoit, M. Wolf, "A well-conditioned estimator for large-dimensional covariance matrices,"
    *J. Multivariate Anal.*, vol. 88, no. 2, pp. 365–411, 2004. Publisher record:
    https://linkinghub.elsevier.com/retrieve/pii/S0047259X03000964 — full text read (published
    version on the author's site): http://www.ledoit.net/Well-conditioned2004.pdf
57. T. M. Cover, J. A. Thomas, *Elements of Information Theory*, 2nd ed., Wiley, 2006.
    https://doi.org/10.1002/047174882X — publisher record and table of contents only (§2.8
    "Data-Processing Inequality", p. 34):
    https://www.wiley.com/en-us/Elements+of+Information+Theory%2C+2nd+Edition-p-9780471241959
58. UNESCO, *Recommendation on the Ethics of Neurotechnology*, adopted by the General Conference at
    its 43rd session, Samarkand, 11 November 2025. **International standard-setting instrument**
    (full text read on the UNESCO legal-instruments page).
    https://www.unesco.org/en/legal-affairs/recommendation-ethics-neurotechnology
59. Colorado General Assembly, HB24-1058, "Protect Privacy of Biological Data," signed 17 April
    2024, effective 7 August 2024. **Statute.** Bill page: https://leg.colorado.gov/bills/hb24-1058
    — signed act read in full: https://leg.colorado.gov/bill_files/41195/download
60. California SB 1223 (Becker, 2023–2024 Regular Session), "Consumer privacy: sensitive personal
    information: neural data." **Statute.** Read as enacted in Cal. Civ. Code §1798.140(ae) and (m),
    California Consumer Privacy Act, consolidated text effective 1 January 2026 published by the
    California Privacy Protection Agency:
    https://cppa.ca.gov/regulations/pdf/ccpa_statute_eff_20260101.pdf ; bill analysis of the
    Assembly Committee on Privacy and Consumer Protection (hearing 2 July 2024) read in full:
    https://apcp.assembly.ca.gov/media/851
61. Regulation (EU) 2016/679 (General Data Protection Regulation), Recital 26, *OJ L 119*,
    4 May 2016. **Statute** (Recital 26 read on EUR-Lex). https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng
62. Personal Information Protection Commission (Korea), 「가명정보 처리 가이드라인」 (Pseudonymised
    information processing guideline), fully revised March 2026. **Government guidance** (full text
    read from the attachment of the press release of 31 March 2026):
    https://www.pipc.go.kr/np/cop/bbs/selectBoardArticle.do?bbsId=BS074&mCode=C020010000&nttId=11928
63. G. Huang, Z. Hu, W. Chen, S. Zhang, Z. Liang, L. Li, L. Zhang, Z. Zhang, "M3CV: A
    multi-subject, multi-session, and multi-task database for EEG-based biometrics challenge,"
    *NeuroImage*, vol. 264, 119666, 2022. https://doi.org/10.1016/j.neuroimage.2022.119666 —
    abstract read (Europe PMC); PubMed record: https://pubmed.ncbi.nlm.nih.gov/36206939/
64. A. K. Chaurasia, M. Fallahi, T. Strufe, P. Terhörst, P. Arias Cabarcos, "NeuroIDBench: An
    open-source benchmark framework for the standardization of methodology in brainwave-based
    authentication research," *J. Inf. Security Appl.*, vol. 85, 103832, 2024.
    https://doi.org/10.1016/j.jisa.2024.103832 — abstract read at https://arxiv.org/abs/2402.08656
65. M. Fallahi, P. Arias-Cabarcos, T. Strufe, "Advancing Brainwave-Based Biometrics: A Large-Scale,
    Multi-Session Evaluation," *IEEE Trans. Biom. Behav. Identity Sci.*, vol. 8, pp. 445–460, 2026.
    https://doi.org/10.1109/TBIOM.2026.3666044 — abstract read at https://arxiv.org/abs/2501.17866
66. H. Ugail, N. Howard, "EEG-Permanence-Benchmark: A Fixed-Protocol Test of Cross-Session EEG
    Biometric Permanence," GitHub repository (README read; accompanying paper listed as under
    review). **Unreviewed repository.** https://github.com/ugail/EEG-Permanence-Benchmark

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

### Section 8 references [39]–[66] (check of 2026-11-12 03:18 KST)

`python scripts/check_references.py` was run again after Section 8 was added, on 2026-11-12
03:18 KST (18:18 UTC on 2026-11-11), and passed; its full output is recorded unchanged in
`results/reference_check/references_20261111T1818Z.txt`. The lines for the identifiers added
by [39]–[66] are reproduced below. Bibliographic fields were compared by hand with the Crossref
record of each DOI, the arXiv record of each preprint, and the standards-body, legislature or
publisher page of the remaining sources. What was read for each reference:

| Ref. | Type | Read | Page checked |
|---|---|---|---|
| [39] ISO/IEC 19795-1:2021 | Standard | Record; preview (contents, clauses 1–6) | IEC webstore; iTeh preview |
| [40] ISO/IEC 24745:2022 | Standard | Record; preview (contents, clauses 1–5) | IEC webstore; iTeh preview |
| [41] Gomez-Barrero et al. 2018 | Journal article | Full text (author version) | Crossref; arXiv:2311.04633 |
| [42]–[44] VoicePrivacy 2020/2022/2024 | Evaluation plans | Full text | arXiv |
| [45] Srivastava et al. 2020 | Conference paper | Full text (author version) | Crossref; arXiv:1911.03934 |
| [46] Brümmer and du Preez 2006 | Journal article | Record page only | Crossref |
| [47] Brümmer 2004 | Workshop paper | Abstract | ISCA Archive |
| [48] Brümmer and du Preez 2013 | Preprint | Full text | arXiv |
| [49] Ramos and Gonzalez-Rodriguez 2013 | Journal article | Abstract | Crossref; PubMed |
| [50] Wald 1945 | Journal article | Record page only | Crossref |
| [51] Ramdas et al. 2023 | Journal article | Full text (author version) | Crossref; arXiv:2210.01948 |
| [52] Grother and Phillips 2004 | Conference paper | Abstract | Crossref; OpenAlex |
| [53] Minka | Technical report | Full text | Author's site |
| [54] Mosimann 1962 | Journal article | Record page only | Crossref |
| [55] Ommen and Saunders 2018 | Journal article | Abstract | Crossref; OpenAlex |
| [56] Ledoit and Wolf 2004 | Journal article | Full text (published version) | Elsevier record; author's site |
| [57] Cover and Thomas 2006 | Book | Record and table of contents only | Crossref; Wiley |
| [58] UNESCO Recommendation 2025 | International instrument | Full text | UNESCO legal-instruments page |
| [59] Colorado HB24-1058 | Statute | Full text (signed act) | Colorado General Assembly |
| [60] California SB 1223 | Statute | Enacted text (consolidated CCPA); Assembly committee analysis | CPPA; California Assembly (leginfo.legislature.ca.gov refused scripted access) |
| [61] GDPR Recital 26 | Statute | Recital 26 | EUR-Lex |
| [62] PIPC guideline 2026 | Government guidance | Full text | PIPC press release and attachment |
| [63] Huang et al. 2022 (M3CV) | Journal article | Abstract | Crossref; Europe PMC; PubMed |
| [64] Chaurasia et al. 2024 (NeuroIDBench) | Journal article | Abstract | Crossref; arXiv:2402.08656 |
| [65] Fallahi et al. 2026 | Journal article | Abstract | Crossref; arXiv:2501.17866 |
| [66] Ugail and Howard | Unreviewed repository | README | GitHub |

```text
reference check, 2026-11-11 18:18 UTC (lines for [39]–[66] only)
  ok   arxiv 1304.2331 -> The PAV algorithm optimizes binary proper scoring rules
  ok   arxiv 1911.03934 -> Evaluating Voice Conversion-based Privacy Protection against Informed Attackers
  ok   arxiv 2203.12468 -> The VoicePrivacy 2022 Challenge Evaluation Plan
  ok   arxiv 2205.07123 -> The VoicePrivacy 2020 Challenge Evaluation Plan
  ok   arxiv 2210.01948 -> Game-theoretic statistics and safe anytime-valid inference
  ok   arxiv 2311.04633 -> General Framework to Evaluate Unlinkability in Biometric Template Protection Systems
  ok   arxiv 2402.08656 -> NeuroIDBench: An Open-Source Benchmark Framework for the Standardization of Methodology in Brainwave-based Authentication Research
  ok   arxiv 2404.02677 -> The VoicePrivacy 2024 Challenge Evaluation Plan
  ok   arxiv 2501.17866 -> Advancing Brainwave-Based Biometrics: A Large-Scale, Multi-Session Evaluation
  ok   doi   10.1002/047174882X -> https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X
  ok   doi   10.1016/j.csl.2005.08.001 -> https://linkinghub.elsevier.com/retrieve/pii/S0885230805000483
  ok   doi   10.1016/j.forsciint.2013.04.014 -> https://linkinghub.elsevier.com/retrieve/pii/S0379073813002375
  ok   doi   10.1016/j.jisa.2024.103832 -> https://linkinghub.elsevier.com/retrieve/pii/S2214212624001340
  ok   doi   10.1016/j.neuroimage.2022.119666 -> https://linkinghub.elsevier.com/retrieve/pii/S105381192200787X
  ok   doi   10.1093/biomet/49.1-2.65 -> https://academic.oup.com/biomet/article-lookup/doi/10.1093/biomet/49.1-2.65
  ok   doi   10.1093/lpr/mgy008 -> https://academic.oup.com/lpr/article/17/2/179/4996009
  ok   doi   10.1109/CVPR.2004.1315146 -> http://ieeexplore.ieee.org/document/1315146/
  ok   doi   10.1109/ICASSP40776.2020.9053868 -> https://ieeexplore.ieee.org/document/9053868/
  ok   doi   10.1109/TBIOM.2026.3666044 -> https://ieeexplore.ieee.org/document/11407492/
  ok   doi   10.1109/TIFS.2017.2788000 -> http://ieeexplore.ieee.org/document/8241848/
  ok   doi   10.1214/23-STS894 -> https://projecteuclid.org/journals/statistical-science/volume-38/issue-4/Game-Theoretic-Statistics-and-Safe-Anytime-Valid-Inference/10.1214/23-STS894.full
  ok   doi   10.1214/aoms/1177731118 -> http://projecteuclid.org/euclid.aoms/1177731118
  ok   url   http://www.ledoit.net/Well-conditioned2004.pdf -> HTTP 200
  ok   url   https://apcp.assembly.ca.gov/media/851 -> HTTP 200
  ok   url   https://cppa.ca.gov/regulations/pdf/ccpa_statute_eff_20260101.pdf -> HTTP 200
  ok   url   https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng -> HTTP 200
  ok   url   https://github.com/ugail/EEG-Permanence-Benchmark -> HTTP 200
  ok   url   https://leg.colorado.gov/bill_files/41195/download -> HTTP 200
  ok   url   https://leg.colorado.gov/bills/hb24-1058 -> HTTP 200
  ok   url   https://linkinghub.elsevier.com/retrieve/pii/S0047259X03000964 -> HTTP 200
  ok   url   https://pubmed.ncbi.nlm.nih.gov/23664798/ -> HTTP 203
  ok   url   https://pubmed.ncbi.nlm.nih.gov/36206939/ -> HTTP 203
  ok   url   https://standards.iteh.ai/catalog/standards/iso/62f63185-3037-442c-8a85-96040c86e5ce/iso-iec-19795-1-2021 -> HTTP 200
  ok   url   https://standards.iteh.ai/catalog/standards/iso/fb018742-972d-4e70-82d3-b8fe3ecc2b3b/iso-iec-24745-2022 -> HTTP 200
  ok   url   https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf -> HTTP 200
  ok   url   https://webstore.iec.ch/en/publication/69063 -> HTTP 200
  ok   url   https://webstore.iec.ch/en/publication/74235 -> HTTP 200
  ok   url   https://www.isca-archive.org/odyssey_2004/brummer04_odyssey.html -> HTTP 200
  ok   url   https://www.pipc.go.kr/np/cop/bbs/selectBoardArticle.do?bbsId=BS074&mCode=C020010000&nttId=11928 -> HTTP 200
  ok   url   https://www.unesco.org/en/legal-affairs/recommendation-ethics-neurotechnology -> HTTP 200
  ok   url   https://www.wiley.com/en-us/Elements+of+Information+Theory%2C+2nd+Edition-p-9780471241959 -> HTTP 200
references: pass
```
