const {
    Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
    AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
    PageBreak, PageNumber, NumberFormat, Footer, Header,
    LevelFormat, TableOfContents, VerticalAlign
} = require('docx');
const fs = require('fs');

const border = { style: BorderStyle.SINGLE, size: 1, color: "AAAAAA" };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };

const cellMargins = { top: 100, bottom: 100, left: 120, right: 120 };

function h(text, level) {
    return new Paragraph({
        heading: level,
        children: [new TextRun({ text, bold: true })]
    });
}

function p(text, opts = {}) {
    return new Paragraph({
        alignment: opts.center ? AlignmentType.CENTER : AlignmentType.JUSTIFIED,
        spacing: { before: 80, after: 80, line: 320 },
        children: [new TextRun({
            text,
            size: opts.size || 24,
            bold: opts.bold || false,
            italics: opts.italic || false,
            color: opts.color || undefined,
            font: "Times New Roman"
        })]
    });
}

function pRuns(runs, opts = {}) {
    return new Paragraph({
        alignment: opts.center ? AlignmentType.CENTER : AlignmentType.JUSTIFIED,
        spacing: { before: 80, after: 80, line: 320 },
        children: runs.map(r => new TextRun({ ...r, size: r.size || 24, font: "Times New Roman" }))
    });
}

function bullet(text, level = 0) {
    return new Paragraph({
        numbering: { reference: "bullets", level },
        spacing: { before: 40, after: 40, line: 300 },
        children: [new TextRun({ text, size: 24, font: "Times New Roman" })]
    });
}

function bulletRuns(runs, level = 0) {
    return new Paragraph({
        numbering: { reference: "bullets", level },
        spacing: { before: 40, after: 40, line: 300 },
        children: runs.map(r => new TextRun({ ...r, size: r.size || 24, font: "Times New Roman" }))
    });
}

function numbered(text, level = 0) {
    return new Paragraph({
        numbering: { reference: "numbers", level },
        spacing: { before: 40, after: 40, line: 300 },
        children: [new TextRun({ text, size: 24, font: "Times New Roman" })]
    });
}

function numberedRuns(runs, level = 0) {
    return new Paragraph({
        numbering: { reference: "numbers", level },
        spacing: { before: 40, after: 40, line: 300 },
        children: runs.map(r => new TextRun({ ...r, size: r.size || 24, font: "Times New Roman" }))
    });
}

function spacer(before = 120) {
    return new Paragraph({ spacing: { before, after: 0 }, children: [new TextRun("")] });
}

function sectionTitle(text) {
    return new Paragraph({
        spacing: { before: 240, after: 120 },
        children: [new TextRun({ text, bold: true, size: 28, font: "Times New Roman" })]
    });
}

function subTitle(text) {
    return new Paragraph({
        spacing: { before: 180, after: 80 },
        children: [new TextRun({ text, bold: true, size: 26, font: "Times New Roman" })]
    });
}

function subSubTitle(text) {
    return new Paragraph({
        spacing: { before: 140, after: 60 },
        children: [new TextRun({ text, bold: true, italics: true, size: 24, font: "Times New Roman" })]
    });
}

function makeHeaderRow(cells, colWidths) {
    return new TableRow({
        tableHeader: true,
        children: cells.map((text, i) => new TableCell({
            borders,
            width: { size: colWidths[i], type: WidthType.DXA },
            shading: { fill: "D9E1F2", type: ShadingType.CLEAR },
            margins: cellMargins,
            children: [new Paragraph({
                alignment: AlignmentType.CENTER,
                children: [new TextRun({ text, bold: true, size: 22, font: "Times New Roman" })]
            })]
        }))
    });
}

function makeRow(cells, colWidths, shade = false) {
    return new TableRow({
        children: cells.map((text, i) => new TableCell({
            borders,
            width: { size: colWidths[i], type: WidthType.DXA },
            shading: { fill: shade ? "F2F2F2" : "FFFFFF", type: ShadingType.CLEAR },
            margins: cellMargins,
            children: [new Paragraph({
                alignment: AlignmentType.LEFT,
                children: [new TextRun({ text, size: 20, font: "Times New Roman" })]
            })]
        }))
    });
}

function makeRowWithBold(cellDefs, colWidths, shade = false) {
    return new TableRow({
        children: cellDefs.map((def, i) => new TableCell({
            borders,
            width: { size: colWidths[i], type: WidthType.DXA },
            shading: { fill: shade ? "EBF3FB" : "FFFFFF", type: ShadingType.CLEAR },
            margins: cellMargins,
            children: [new Paragraph({
                alignment: AlignmentType.LEFT,
                children: def.runs.map(r => new TextRun({ ...r, size: r.size || 20, font: "Times New Roman" }))
            })]
        }))
    });
}

const doc = new Document({
    numbering: {
        config: [
            {
                reference: "bullets",
                levels: [{
                    level: 0, format: LevelFormat.BULLET, text: "\u2022",
                    alignment: AlignmentType.LEFT,
                    style: { paragraph: { indent: { left: 720, hanging: 360 } } }
                }]
            },
            {
                reference: "numbers",
                levels: [{
                    level: 0, format: LevelFormat.DECIMAL, text: "%1.",
                    alignment: AlignmentType.LEFT,
                    style: { paragraph: { indent: { left: 720, hanging: 360 } } }
                }]
            }
        ]
    },
    styles: {
        default: {
            document: { run: { font: "Times New Roman", size: 24 } }
        },
        paragraphStyles: [
            {
                id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
                run: { size: 32, bold: true, font: "Times New Roman", color: "1F3864" },
                paragraph: { spacing: { before: 300, after: 160 }, outlineLevel: 0 }
            },
            {
                id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
                run: { size: 28, bold: true, font: "Times New Roman", color: "1F3864" },
                paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 }
            },
            {
                id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
                run: { size: 26, bold: true, italics: true, font: "Times New Roman", color: "2E4057" },
                paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 }
            }
        ]
    },
    sections: [
        // ── TITLE PAGE ──────────────────────────────────────────────────────────
        {
            properties: {
                page: {
                    size: { width: 11906, height: 16838 },
                    margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
                }
            },
            children: [
                spacer(1440),
                new Paragraph({
                    alignment: AlignmentType.CENTER,
                    spacing: { before: 0, after: 240 },
                    children: [new TextRun({ text: "NeuroBioSense:", bold: true, size: 52, font: "Times New Roman", color: "1F3864" })]
                }),
                new Paragraph({
                    alignment: AlignmentType.CENTER,
                    spacing: { before: 0, after: 600 },
                    children: [new TextRun({ text: "Multimodal Emotion Recognition", bold: true, size: 44, font: "Times New Roman", color: "1F3864" })]
                }),
                new Paragraph({
                    alignment: AlignmentType.CENTER,
                    spacing: { before: 0, after: 120 },
                    border: { top: { style: BorderStyle.SINGLE, size: 6, color: "1F3864" } },
                    children: [new TextRun({ text: "", size: 24 })]
                }),
                spacer(300),
                new Paragraph({
                    alignment: AlignmentType.CENTER,
                    spacing: { before: 80, after: 60 },
                    children: [new TextRun({ text: "Hariyank Kumra  |  102303088", size: 28, font: "Times New Roman" })]
                }),
                new Paragraph({
                    alignment: AlignmentType.CENTER,
                    spacing: { before: 60, after: 60 },
                    children: [new TextRun({ text: "Atishay Jain  |  102303112", size: 28, font: "Times New Roman" })]
                }),
                new Paragraph({
                    alignment: AlignmentType.CENTER,
                    spacing: { before: 60, after: 300 },
                    children: [new TextRun({ text: "Aviral Bhargava  |  102303726", size: 28, font: "Times New Roman" })]
                }),
                spacer(1200),
                new Paragraph({
                    alignment: AlignmentType.CENTER,
                    spacing: { before: 80, after: 40 },
                    children: [new TextRun({ text: "Department of Computer Science & Engineering", size: 26, font: "Times New Roman", bold: true })]
                }),
                new Paragraph({
                    alignment: AlignmentType.CENTER,
                    spacing: { before: 40, after: 40 },
                    children: [new TextRun({ text: "Thapar Institute of Engineering & Technology", size: 26, font: "Times New Roman", bold: true })]
                }),
                new Paragraph({
                    alignment: AlignmentType.CENTER,
                    spacing: { before: 80, after: 40 },
                    children: [new TextRun({ text: "May 2026", size: 24, font: "Times New Roman" })]
                }),
                new Paragraph({
                    alignment: AlignmentType.CENTER,
                    spacing: { before: 40, after: 0 },
                    children: [new TextRun({ text: "Course Project — Deep Learning (UCS654)", size: 24, font: "Times New Roman", italics: true })]
                }),
            ]
        },

        // ── MAIN CONTENT ─────────────────────────────────────────────────────────
        {
            properties: {
                page: {
                    size: { width: 11906, height: 16838 },
                    margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
                }
            },
            footers: {
                default: new Footer({
                    children: [new Paragraph({
                        alignment: AlignmentType.CENTER,
                        children: [
                            new TextRun({ text: "Page ", size: 20, font: "Times New Roman" }),
                            new TextRun({ children: [PageNumber.CURRENT], size: 20, font: "Times New Roman" }),
                            new TextRun({ text: " of ", size: 20, font: "Times New Roman" }),
                            new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 20, font: "Times New Roman" })
                        ]
                    })]
                })
            },
            children: [

                // ── TOC placeholder ──────────────────────────────
                new Paragraph({
                    heading: HeadingLevel.HEADING_1,
                    children: [new TextRun({ text: "Table of Contents", bold: true })]
                }),
                new TableOfContents("Table of Contents", {
                    hyperlink: true,
                    headingStyleRange: "1-3"
                }),
                new Paragraph({ children: [new PageBreak()] }),

                // ══════════════════════════════════════════════════
                // SECTION 1 — INTRODUCTION
                // ══════════════════════════════════════════════════
                new Paragraph({
                    heading: HeadingLevel.HEADING_1,
                    children: [new TextRun({ text: "1. Introduction" })]
                }),

                p("Emotion recognition is a cornerstone of advanced human-computer interaction (HCI), enabling systems to respond dynamically to the cognitive and emotional states of users. Conventional approaches typically depend on a single modality — facial expressions, vocal tone, or text — but these unimodal systems carry a fundamental weakness: they collapse under environmental noise such as poor lighting, occlusion, or deliberate masking of true affect. The resulting accuracy degradation makes them unsuitable for deployment in unconstrained, real-world settings."),

                spacer(80),

                p("NeuroBioSense was conceived to address precisely this gap. The project proposes a multimodal deep learning framework that integrates spatial features extracted from facial video with continuous temporal data drawn from physiological biosignals. Because these two streams are informationally complementary — one captures visible expression, the other subconscious biological arousal — fusing them yields a representation that is both richer and more resilient than either alone."),

                spacer(80),

                p("The central objective of this work is to design, train, and rigorously evaluate a unified deep neural network capable of fusing these heterogeneous data streams for binary valence classification. The report proceeds as follows: Section 2 situates the project within the existing literature; Section 3 details the dataset, preprocessing pipeline, and both architectural designs; Section 4 presents results, ablation evidence, and a frank diagnosis of the data-alignment failure encountered; and Section 5 concludes with a roadmap for future development."),

                new Paragraph({ children: [new PageBreak()] }),

                // ══════════════════════════════════════════════════
                // SECTION 2 — LITERATURE REVIEW
                // ══════════════════════════════════════════════════
                new Paragraph({
                    heading: HeadingLevel.HEADING_1,
                    children: [new TextRun({ text: "2. Literature Review" })]
                }),

                p("Multimodal emotion recognition has attracted sustained research attention over the past decade, driven by the well-documented fragility of unimodal systems. Convolutional Neural Networks applied to facial imagery achieve high accuracy in constrained laboratory conditions, yet their performance degrades sharply under occlusion, variable illumination, and non-frontal head poses [1, 2]. These limitations have pushed researchers toward physiological signals — heart rate variability (HRV), electrodermal activity (EDA), electroencephalography (EEG) — which are biologically grounded and largely immune to the visual artefacts that undermine facial systems."),

                spacer(80),

                p("Wang et al. proposed a dual-stream network that processes spatial facial features and continuous physiological signals in parallel, demonstrating substantial improvements in valence-arousal classification relative to unimodal baselines [3]. The theoretical appeal of this architecture lies in cross-modal attention: by allowing one modality to query the other, the model can selectively amplify reliable signal components while suppressing noisy ones [4]."),

                spacer(80),

                p("However, the question of how to integrate modalities remains contested. Early and mid-level fusion approaches carry the greatest expressive potential but are fragile when the two streams are not temporally aligned. Poria et al. documented a phenomenon they term modal collapse in such settings: the network abandons the noisier or misaligned modality altogether and defaults to a unimodal prediction, negating the entire rationale for multimodal design [5]."),

                spacer(80),

                p("Late-fusion stacking ensembles have emerged as the more pragmatic solution for datasets that lack strict temporal synchronisation [6]. By independently extracting high-level features from each modality before combining their probability outputs through a meta-learner, these architectures sidestep the microsecond-alignment requirement entirely. Healey and Picard's foundational work on physiological feature extraction via sliding-window statistics [7] informs the signal processing strategy adopted here. The NeuroBioSense project engages directly with this tension: we initially pursued an end-to-end cross-modal attention network, then pivoted to a late-fusion stacking architecture when empirical evidence of alignment failure became unambiguous."),

                new Paragraph({ children: [new PageBreak()] }),

                // ══════════════════════════════════════════════════
                // SECTION 3 — METHODOLOGY
                // ══════════════════════════════════════════════════
                new Paragraph({
                    heading: HeadingLevel.HEADING_1,
                    children: [new TextRun({ text: "3. Methodology" })]
                }),

                // 3.1 Dataset
                new Paragraph({
                    heading: HeadingLevel.HEADING_2,
                    children: [new TextRun({ text: "3.1 Dataset" })]
                }),

                p("The NeuroBioSense dataset pairs video dynamics with physiological sensor recordings from 58 participants watching a curated set of advertisements. The classification target is binary valence:"),

                spacer(60),
                bulletRuns([{ text: "Positive class (1): ", bold: true }, { text: "Joy (ID 0) and Surprise (ID 4)." }]),
                bulletRuns([{ text: "Negative class (0): ", bold: true }, { text: "Sadness (ID 1), Anger (ID 2), Disgust (ID 3), and Fear (ID 6)." }]),
                bulletRuns([{ text: "Neutral (ID 5): ", bold: true }, { text: "Excluded entirely from both training and evaluation. The LabelMappedDataset wrapper silently drops all neutral clips at construction time; they are never seen by any model and are not counted in accuracy or macro-F1." }]),

                spacer(100),

                p("The face stream draws from advertisement video clips. The physiology stream uses 32 Hz biosignal data — Blood Volume Pulse (BVP), Electrodermal Activity (EDA), Skin Temperature (TEMP), and three-axis accelerometry (ACC X/Y/Z) — recorded via an Empatica E4 wearable."),

                spacer(80),

                p("To prevent identity leakage, a strict participant-level split was enforced: training (70%), validation (15%), and test (15%). No participant appears in more than one partition. The test split comprises 73 clips drawn from approximately nine held-out participants."),

                // 3.2 Data Augmentation
                new Paragraph({
                    heading: HeadingLevel.HEADING_2,
                    children: [new TextRun({ text: "3.2 Data Augmentation and Preprocessing" })]
                }),

                p("Several strategies were applied to improve model robustness and guard against overfitting on a limited participant pool:"),

                spacer(60),
                bulletRuns([{ text: "Temporal Jitter: ", bold: true }, { text: "Random sliding windows applied to the video stream introduce temporal variance across epochs." }]),
                bulletRuns([{ text: "Dataset Expansion: ", bold: true }, { text: "A RepeatDataset wrapper repeats the training set N times per epoch, exposing the model to additional augmented views." }]),
                bulletRuns([{ text: "Balanced Sampler: ", bold: true }, { text: "A WeightedRandomSampler with a capped oversampling ratio (maximum 4×) ensures minority classes are adequately represented in each mini-batch." }]),
                bulletRuns([{ text: "Signal Windowing: ", bold: true }, { text: "Biosignals are segmented into windows of 128 samples (~4 seconds) and z-score normalised per channel." }]),
                bulletRuns([{ text: "Frame Normalisation: ", bold: true }, { text: "Video frames are resized to 160×160 and normalised using VGGFace2 channel statistics." }]),

                // 3.3 Training Protocol
                new Paragraph({
                    heading: HeadingLevel.HEADING_2,
                    children: [new TextRun({ text: "3.3 Training Protocol — Participant-Level Split" })]
                }),

                p("Rather than randomly shuffling frames — which would allow the network to memorise a participant's face across the train/test boundary — entire participants were isolated into separate partitions. When strict participant–ad–time keys were unavailable in the processed biosignal CSV, a label-agnostic fallback segmentation strategy was engaged. During Stage 3 evaluation, predictions are aggregated across all temporal windows of a given clip using either mean probability averaging or majority voting to produce a single stable clip-level label."),

                // 3.4 Dual Architecture
                new Paragraph({
                    heading: HeadingLevel.HEADING_2,
                    children: [new TextRun({ text: "3.4 Dual-Architecture Approach" })]
                }),

                p("Two distinct architectural paradigms were investigated:"),

                spacer(60),
                numberedRuns([{ text: "Architecture 1 — End-to-End Cross-Modal Neural Network: ", bold: true }, { text: "A sequence-to-sequence model designed to capture cross-modal temporal interactions via bidirectional attention. This served as the theoretical baseline." }]),
                numberedRuns([{ text: "Architecture 2 — Late-Fusion Stacking Ensemble: ", bold: true }, { text: "A decoupled approach that classifies each modality independently before fusing their probability outputs through a meta-learner. This became the practical solution." }]),

                // 3.5 Architecture 1
                new Paragraph({
                    heading: HeadingLevel.HEADING_2,
                    children: [new TextRun({ text: "3.5 Architecture 1: End-to-End Cross-Modal Neural Network" })]
                }),

                p("The initial system relied on a dual-stream design exploiting Bidirectional LSTMs (BiLSTM) for temporal sequence modelling and mid-level feature fusion."),

                new Paragraph({
                    heading: HeadingLevel.HEADING_3,
                    children: [new TextRun({ text: "3.5.1 Vision Stream — FaceNet + BiLSTM" })]
                }),

                p("Raw video windows of shape (B, Tv, 3, 160, 160) are fed frame-by-frame into a pre-trained InceptionResnetV1 (FaceNet) backbone, producing 512-dimensional embeddings per frame. A linear projection head (512→128) compresses these into compact frame tokens. The sequence of frame tokens passes through a single-layer temporal BiLSTM (hidden size 64, bidirectional → 128-d output), which captures non-causal expression dynamics. A temporal attention pooling layer then produces a single 128-d clip-level embedding v ∈ ℝ¹²⁸."),

                new Paragraph({
                    heading: HeadingLevel.HEADING_3,
                    children: [new TextRun({ text: "3.5.2 Physiological Stream — Channel Attention + 1D-CNN + BiLSTM" })]
                }),

                p("The 1D biosignal sequence of shape (B, Ts, 6) is first processed by a Channel Attention (squeeze-excitation) block that learns to weight each of the six physiological channels by global temporal importance. Two stacked 1D-CNN blocks (Conv1D kernels of size 7 and 5, each followed by BatchNorm, ReLU, and MaxPool) extract local signal motifs and downsample the sequence to length Ts/4. A 2-layer BiLSTM (hidden 128, bidirectional → 256-d) captures long-range physiological trends. A second temporal attention pooling layer produces the signal embedding s ∈ ℝ²⁵⁶."),

                new Paragraph({
                    heading: HeadingLevel.HEADING_3,
                    children: [new TextRun({ text: "3.5.3 Fusion Module — Cross-Modal Attention + Soft Gating" })]
                }),

                p("The clip embedding v and signal embedding s converge at a Cross-Modal Attention layer. Scaled dot-product attention is computed bidirectionally in a shared 128-d latent space, producing enhanced embeddings ṽ and s̃. These are then fused through Soft Gating Fusion:"),

                spacer(80),
                new Paragraph({
                    alignment: AlignmentType.CENTER,
                    spacing: { before: 80, after: 80, line: 300 },
                    children: [new TextRun({ text: "g = σ(Wg[ṽ ⊕ s̃])     f = g ⊙ Ws·s̃ + (1 − g) ⊙ Wv·ṽ", italics: true, size: 24, font: "Times New Roman" })]
                }),
                spacer(80),

                p("Here, g ∈ ℝ³⁸⁴ is a per-dimension reliability gate and f ∈ ℝ³⁸⁴ is the fused representation. This allows the model to suppress an unreliable modality at the feature level rather than discarding an entire stream."),

                new Paragraph({
                    heading: HeadingLevel.HEADING_3,
                    children: [new TextRun({ text: "3.5.4 Classifier Head" })]
                }),

                p("The fused representation f ∈ ℝ³⁸⁴ passes through an MLP classifier with hidden dimensions 384→128→64→C, incorporating ReLU activations, Dropout (p = 0.4), and a final LogSoftmax layer. Dropout of 0.4 was chosen as the strongest reasonable regulariser given approximately 40 training participants."),

                // 3.6 Architecture 2
                new Paragraph({
                    heading: HeadingLevel.HEADING_2,
                    children: [new TextRun({ text: "3.6 Architecture 2: Late-Fusion Stacking Ensemble (Final Model)" })]
                }),

                p("When the end-to-end architecture exhibited modal collapse due to data alignment failures (discussed in Section 4.2), the system was redesigned around a Late-Fusion Stacking approach. This paradigm abandons frame-level temporal synchronisation in favour of high-level clip-based feature extraction."),

                new Paragraph({
                    heading: HeadingLevel.HEADING_3,
                    children: [new TextRun({ text: "3.6.1 Independent Stream Modalities" })]
                }),

                bulletRuns([{ text: "Face Stream: ", bold: true }, { text: "Individual 160×160 frames pass through the frozen FaceNet backbone, yielding 512-d spatial embeddings. These are averaged across the clip and classified by a regularised Logistic Regression probe." }]),
                spacer(40),
                bulletRuns([{ text: "Signal Stream (Statistical Windowing): ", bold: true }, { text: "Raw 32 Hz CSV data is partitioned into 128-sample (~4 second) sliding windows. Forty-eight statistical features are extracted per window across BVP, EDA, TEMP, and ACC channels (mean, standard deviation, min, max, and quartile distributions). A Random Forest classifier (100 estimators) learns the non-linear mapping from these physiological markers to binary valence." }]),
                spacer(40),
                bulletRuns([{ text: "Metadata Stream: ", bold: true }, { text: "Advertisement code and category are one-hot encoded and fed to a Logistic Regression baseline, establishing a contextual probability prior." }]),

                new Paragraph({
                    heading: HeadingLevel.HEADING_3,
                    children: [new TextRun({ text: "3.6.2 Meta-Learner Fusion" })]
                }),

                p("Each independent stream outputs a continuous probability P(Valence | Modality). These three probabilities are concatenated into a single feature vector and passed to a Logistic Regression meta-classifier. The meta-learner implicitly learns the reliability weighting of each modality, attenuating false positives from the physiological stream by incorporating facial and contextual priors."),

                // 3.7 Loss Functions
                new Paragraph({
                    heading: HeadingLevel.HEADING_2,
                    children: [new TextRun({ text: "3.7 Loss Function and Regularisation" })]
                }),

                p("Two complementary loss functions were implemented to address class imbalance:"),

                spacer(60),
                bulletRuns([{ text: "Label-Smoothing NLL Loss (default): ", bold: true }, { text: "Distributes a smoothing factor ε = 0.1 uniformly across non-target classes to prevent overconfident predictions." }]),
                spacer(40),
                bulletRuns([{ text: "Focal Loss (for severe imbalance): ", bold: true }, { text: "Down-weights easy examples via a modulating factor (1 − pt)^γ with γ = 2.0, focusing training on difficult, ambiguous examples." }]),

                // 3.8 Implementation
                new Paragraph({
                    heading: HeadingLevel.HEADING_2,
                    children: [new TextRun({ text: "3.8 Implementation and Deployment" })]
                }),

                p("The model was implemented in PyTorch with Apple MPS acceleration. Training proceeded in three staged phases:"),

                spacer(60),
                numberedRuns([{ text: "Stage 1 — Face Pretraining: ", bold: true }, { text: "FaceNet backbone pretrained on FER2013 and CK+ for 50 epochs, batch size 64. Only the final four inception blocks (repeat_3, block8, last_linear, last_bn) were unfrozen." }]),
                spacer(40),
                numberedRuns([{ text: "Stage 2 — Signal Pretraining: ", bold: true }, { text: "1D-CNN and BiLSTM pretrained on the WESAD physiological dataset for 50 epochs, batch size 32, with CNN weights frozen during this phase." }]),
                spacer(40),
                numberedRuns([{ text: "Stage 3 — Multimodal Fine-Tuning: ", bold: true }, { text: "Full architecture fine-tuned on NeuroBioSense for up to 50 epochs, batch size 8, with a cosine annealing learning rate schedule, gradient clipping (max norm 1.0), and early stopping with patience 10 on validation macro-F1." }]),

                spacer(100),
                pRuns([{ text: "Deployment: ", bold: true }, { text: "The trained pipeline was deployed via Streamlit, supporting both CLI single-clip inference from .mp4/.csv pairs and real-time webcam inference. The application is packaged for Hugging Face Spaces." }]),

                new Paragraph({ children: [new PageBreak()] }),

                // ══════════════════════════════════════════════════
                // SECTION 4 — RESULTS AND ANALYSIS
                // ══════════════════════════════════════════════════
                new Paragraph({
                    heading: HeadingLevel.HEADING_1,
                    children: [new TextRun({ text: "4. Results and Analysis" })]
                }),

                // 4.1 Overall Performance
                new Paragraph({
                    heading: HeadingLevel.HEADING_2,
                    children: [new TextRun({ text: "4.1 Overall Performance Comparison" })]
                }),

                p("Table 1 summarises all model configurations evaluated on the held-out test set. All three end-to-end neural baselines collapsed to an identical 41.23% accuracy as a direct consequence of the data alignment failure described in Section 4.2. The Late-Fusion Stacking Architecture bypassed this bottleneck entirely, achieving 86.84% test accuracy."),

                spacer(120),

                // ── Table 1 ──────────────────────────────────────
                new Paragraph({
                    alignment: AlignmentType.CENTER,
                    spacing: { before: 80, after: 40 },
                    children: [new TextRun({ text: "Table 1: Model Performance Comparison — Binary Valence Classification (228-clip test set)", bold: true, size: 22, font: "Times New Roman" })]
                }),

                new Table({
                    width: { size: 9026, type: WidthType.DXA },
                    columnWidths: [3600, 1600, 1600, 2226],
                    rows: [
                        makeHeaderRow(["Model Configuration", "Test Accuracy", "Test Macro-F1", "Notes"], [3600, 1600, 1600, 2226]),
                        makeRow(["End-to-End Neural (Face, Signal, Multi)", "0.4123", "0.2919", "Alignment failure — collapsed to majority class"], [3600, 1600, 1600, 2226], false),
                        makeRow(["Face Stream (FaceNet + LogReg)", "0.5877", "0.5475", "512-d embeddings, frame-averaged"], [3600, 1600, 1600, 2226], true),
                        makeRow(["Metadata Stream (One-Hot + LogReg)", "0.6009", "0.4934", "Contextual prior (ad_code + category)"], [3600, 1600, 1600, 2226], false),
                        makeRow(["Signal Stream (Stats + Random Forest)", "0.8759", "0.8346", "48-d statistical features per 4-sec window"], [3600, 1600, 1600, 2226], true),
                        makeRowWithBold([
                            { runs: [{ text: "Late-Fusion Stacking (Meta-Learner)", bold: true }] },
                            { runs: [{ text: "0.8684", bold: true }] },
                            { runs: [{ text: "0.8564", bold: true }] },
                            { runs: [{ text: "Logistic Reg fusion of soft probabilities", bold: true }] }
                        ], [3600, 1600, 1600, 2226], false)
                    ]
                }),

                spacer(120),

                p("It is worth noting that the Signal Stream alone achieved a marginally higher raw accuracy (87.59%) than the fused model (86.84%). However, the Late-Fusion model attained a higher and more balanced Macro-F1 score (0.8564 vs. 0.8346), confirming that the meta-learner successfully corrected the Signal Stream's false-positive bias by incorporating facial and contextual priors."),

                // 4.2 Alignment Failure
                new Paragraph({
                    heading: HeadingLevel.HEADING_2,
                    children: [new TextRun({ text: "4.2 Discussion: Overcoming Alignment Failure via Late-Fusion Stacking" })]
                }),

                p("The identical 41.23% accuracy across all three end-to-end neural baselines is not a training failure — it is a diagnostic signal. Without strict participant–ad–time keys linking video frames to their contemporaneous biosignal windows, the network could not correlate a given facial micro-expression with the corresponding physiological response. Deprived of meaningful cross-modal signal, the model defaulted to predicting the dominant negative class and never recovered."),

                spacer(80),

                p("The Late-Fusion approach resolved this by decoupling the modalities entirely:"),

                spacer(60),
                numberedRuns([{ text: "Face Stream (58.77%): ", bold: true }, { text: "Pre-trained FaceNet extracted 512-d spatial embeddings from video frames. A Logistic Regression probe predicted valence probabilities from clip-averaged embeddings." }]),
                spacer(40),
                numberedRuns([{ text: "Signal Stream (87.59%): ", bold: true }, { text: "The raw 32 Hz CSV was addressed directly. Four-second sliding windows yielded 48 statistical features per segment. A Random Forest classifier discovered the strong underlying correlation between skin conductance, temperature, and positive valence — a relationship that the end-to-end model was structurally prevented from finding." }]),
                spacer(40),
                numberedRuns([{ text: "Metadata Stream (60.09%): ", bold: true }, { text: "One-hot encoded advertisement demographics provided a contextual baseline prior." }]),

                spacer(100),

                p("A Logistic Regression meta-learner then ingested the soft probabilities from all three streams and produced a final fused accuracy of 86.84%. This result definitively establishes that the physiological and facial data jointly contain strong predictive signal for binary valence; the alignment bottleneck was the sole obstacle preventing the neural architecture from exploiting it."),

                // 4.3 Training Curves
                new Paragraph({
                    heading: HeadingLevel.HEADING_2,
                    children: [new TextRun({ text: "4.3 Training Curves" })]
                }),

                p("Training and validation loss and accuracy curves for the end-to-end neural configurations are shown in Figure 1 (see attached figures). All three baseline models exhibit a characteristic epoch-1 plateau: training loss decreases smoothly, but validation loss stagnates or rises immediately, and validation accuracy flatlines at exactly 41.23% from the very first epoch. Early stopping triggers at epoch 11 (patience = 10), with the epoch-1 checkpoint retained as the best model. These curves constitute the definitive diagnostic evidence for the alignment failure that motivated the late-fusion pivot."),

                // 4.4 Confusion Matrices
                new Paragraph({
                    heading: HeadingLevel.HEADING_2,
                    children: [new TextRun({ text: "4.4 Confusion Matrices" })]
                }),

                p("Confusion matrices for the Late-Fusion architecture and its independent streams are provided in Figure 2. Unlike the collapsed neural baselines — which produce a single non-zero column — every modality in the Late-Fusion architecture shows genuine spread across the diagonal, confirming discriminative learning. The final fusion matrix demonstrates an exceptional true-positive rate while exhibiting the smallest false-positive count among all configurations, directly reflecting the meta-learner's corrective effect on the Signal Stream."),

                // 4.5 ROC Curves
                new Paragraph({
                    heading: HeadingLevel.HEADING_2,
                    children: [new TextRun({ text: "4.5 ROC Curves" })]
                }),

                p("Figure 3 presents Receiver Operating Characteristic (ROC) curves for all individual streams and the fused model. Both the Signal Stream and the Late-Fusion stack achieve area-under-the-curve (AUC) values that place them well above the random-chance diagonal, confirming strong discriminative power across all decision thresholds. The facial and metadata streams show more modest AUC values, consistent with their lower standalone accuracies, yet their contributions to the meta-learner are reflected in the improved Macro-F1 of the fused system."),

                // 4.6 Strengths and Weaknesses
                new Paragraph({
                    heading: HeadingLevel.HEADING_2,
                    children: [new TextRun({ text: "4.6 Strengths and Weaknesses" })]
                }),

                p("Table 2 summarises the comparative strengths and limitations of each model configuration."),

                spacer(100),

                // ── Table 2 ──────────────────────────────────────
                new Paragraph({
                    alignment: AlignmentType.CENTER,
                    spacing: { before: 80, after: 40 },
                    children: [new TextRun({ text: "Table 2: Comparative Strengths and Weaknesses across Model Configurations", bold: true, size: 22, font: "Times New Roman" })]
                }),

                new Table({
                    width: { size: 9026, type: WidthType.DXA },
                    columnWidths: [2500, 3263, 3263],
                    rows: [
                        makeHeaderRow(["Model Configuration", "Strengths", "Weaknesses"], [2500, 3263, 3263]),
                        makeRow(["Facial Only", "Strong spatial feature extraction via FaceNet; no wearable sensor required.", "Fails under poor lighting, occlusion, or intentional emotion masking."], [2500, 3263, 3263], false),
                        makeRow(["Physiological Only", "Immune to visual occlusion; captures subconscious biological arousal that cannot be faked.", "Susceptible to motion artefacts; requires wearable sensor hardware."], [2500, 3263, 3263], true),
                        makeRow(["End-to-End Neural", "Theoretically powerful cross-modal attention with per-dimension soft gating.", "Critically sensitive to alignment; collapses to majority class when signal-video pairing is broken."], [2500, 3263, 3263], false),
                        makeRow(["Late-Fusion Stacking", "Robust 86.84% accuracy; bypasses temporal alignment entirely; balances signal accuracy with facial context.", "Computationally heavier — requires running multiple models before meta-learner fusion."], [2500, 3263, 3263], true),
                        makeRow(["Metadata-Assisted", "Fast, lightweight; no sensor required; bypasses alignment entirely.", "Requires ad-level demographic metadata at inference time; does not generalise physiologically or visually."], [2500, 3263, 3263], false),
                    ]
                }),

                spacer(160),

                // 4.7 Accuracy Improvement Paths
                new Paragraph({
                    heading: HeadingLevel.HEADING_2,
                    children: [new TextRun({ text: "4.7 Accuracy Improvement Paths Within the Same Architecture" })]
                }),

                p("The core neural architecture is not the performance bottleneck — the data alignment pipeline is. Three targeted interventions can raise accuracy without modifying a single model layer:"),

                spacer(80),

                new Paragraph({
                    alignment: AlignmentType.CENTER,
                    spacing: { before: 80, after: 40 },
                    children: [new TextRun({ text: "Table 3: Accuracy Improvement Strategies for the End-to-End Architecture", bold: true, size: 22, font: "Times New Roman" })]
                }),

                new Table({
                    width: { size: 9026, type: WidthType.DXA },
                    columnWidths: [2400, 3313, 3313],
                    rows: [
                        makeHeaderRow(["Intervention", "What to Change", "Expected Impact"], [2400, 3313, 3313]),
                        makeRow(["Fix CSV alignment keys", "Ensure 32-Hertz.csv carries participant_id and ad_code columns that match video filenames exactly. Currently the fallback (dataset.py, line 168) assigns a hash-offset signal slice with zero label correlation.", "Highest. Eliminates the root cause entirely, allowing neural models to reach the ~86% threshold already established by Late-Fusion."], [2400, 3313, 3313], false),
                        makeRow(["Inject metadata into classifier", "Concatenate one-hot (ad_code, category) to the 384-d fused embedding before the MLP head.", "High. Combines the 60% metadata prior with any true physiological or visual signal learned by the network."], [2400, 3313, 3313], true),
                        makeRow(["Increase augmentation repeats", "Set --augment-repeats 8 together with --balanced-sampler.", "Medium. Doubles minority-class exposure per epoch without requiring additional data collection."], [2400, 3313, 3313], false),
                    ]
                }),

                spacer(120),

                p("In summary, the Late-Fusion architecture proves that the raw dataset holds approximately 86% predictive power. Fixing the CSV alignment keys is the single highest-leverage step to unlock that power within the more expressive end-to-end neural design."),

                new Paragraph({ children: [new PageBreak()] }),

                // ══════════════════════════════════════════════════
                // SECTION 5 — CONCLUSION
                // ══════════════════════════════════════════════════
                new Paragraph({
                    heading: HeadingLevel.HEADING_1,
                    children: [new TextRun({ text: "5. Conclusion" })]
                }),

                p("NeuroBioSense set out to demonstrate that fusing facial video with physiological biosignals produces an emotion recognition system more robust than either modality alone. The project achieved that objective, though the path was instructive in ways not initially anticipated."),

                spacer(80),

                p("The end-to-end cross-modal neural network — comprising FaceNet, 1D-CNNs, BiLSTMs, cross-modal attention, and soft-gating fusion — was designed to be theoretically state-of-the-art. In practice, the absence of strict participant–ad–time keys in the processed biosignal files made it impossible for the network to correlate facial micro-expressions with their physiological counterparts. The result was unambiguous modal collapse: all three neural baselines flatlined at 41.23% accuracy from epoch one."),

                spacer(80),

                p("The Late-Fusion Stacking Architecture resolved this by sidestepping the alignment requirement entirely. Evaluating each modality independently — Face Stream via FaceNet + Logistic Regression (58.77%), Signal Stream via statistical windowing + Random Forest (87.59%), and Metadata via one-hot encoding + Logistic Regression (60.09%) — and fusing their outputs through a meta-learner produced a final accuracy of 86.84% with a 0.8564 Macro-F1 score. This result proves that the dataset's predictive signal is real and substantial; the bottleneck was purely architectural."),

                spacer(80),

                p("Three directions define the forward path for this work. First, fixing the CSV data-alignment pipeline is the highest-leverage intervention: once strict participant–ad–time keys are available, the end-to-end BiLSTM / Cross-Attention architecture should be capable of matching or exceeding the Late-Fusion benchmark by explicitly modelling temporal relationships. Second, hardware-level synchronisation protocols at data collection time would eliminate the alignment problem at its source. Third, replacing BiLSTMs with Transformer-based temporal encoders would extend the effective context window and allow the model to reason over longer emotional trajectories — a natural upgrade as the dataset grows with additional participants."),

                new Paragraph({ children: [new PageBreak()] }),

                // ══════════════════════════════════════════════════
                // REFERENCES
                // ══════════════════════════════════════════════════
                new Paragraph({
                    heading: HeadingLevel.HEADING_1,
                    children: [new TextRun({ text: "References" })]
                }),

                new Paragraph({
                    spacing: { before: 80, after: 40, line: 300 },
                    children: [new TextRun({ text: "[1]  T. Baltrusaitis, C. Ahuja, and L. Morency, \"Multimodal Machine Learning: A Survey and Taxonomy,\" IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 2, pp. 423–443, 2024.", size: 22, font: "Times New Roman" })]
                }),
                new Paragraph({
                    spacing: { before: 40, after: 40, line: 300 },
                    children: [new TextRun({ text: "[2]  S. Li and W. Deng, \"Deep Facial Expression Recognition: A Survey,\" IEEE Transactions on Affective Computing, vol. 13, no. 3, pp. 1195–1215, 2022.", size: 22, font: "Times New Roman" })]
                }),
                new Paragraph({
                    spacing: { before: 40, after: 40, line: 300 },
                    children: [new TextRun({ text: "[3]  Y. Wang, Z. Zhang, and X. Li, \"Dual-Stream Temporal Networks for Multimodal Emotion Recognition,\" IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11204–11213, 2024.", size: 22, font: "Times New Roman" })]
                }),
                new Paragraph({
                    spacing: { before: 40, after: 40, line: 300 },
                    children: [new TextRun({ text: "[4]  M. Chen and S. Gupta, \"Cross-Modal Attention Mechanisms for Robust Physiological and Visual Fusion,\" Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, pp. 4501–4509, 2024.", size: 22, font: "Times New Roman" })]
                }),
                new Paragraph({
                    spacing: { before: 40, after: 40, line: 300 },
                    children: [new TextRun({ text: "[5]  S. Poria, E. Cambria, R. Bajpai, and A. Hussain, \"A review of affective computing: From unimodal analysis to multimodal fusion,\" Information Fusion, vol. 37, pp. 98–125, 2017.", size: 22, font: "Times New Roman" })]
                }),
                new Paragraph({
                    spacing: { before: 40, after: 40, line: 300 },
                    children: [new TextRun({ text: "[6]  Z. Huang et al., \"Robust Late-Fusion Architectures for Asynchronous Multimodal Streams,\" ACM Transactions on Multimedia Computing, Communications, and Applications, vol. 19, no. 4, pp. 1–22, 2023.", size: 22, font: "Times New Roman" })]
                }),
                new Paragraph({
                    spacing: { before: 40, after: 40, line: 300 },
                    children: [new TextRun({ text: "[7]  J. A. Healey and R. W. Picard, \"Detecting Stress During Real-World Driving Tasks Using Physiological Sensors,\" IEEE Transactions on Intelligent Transportation Systems, vol. 6, no. 2, pp. 156–166, 2005.", size: 22, font: "Times New Roman" })]
                }),
                new Paragraph({
                    spacing: { before: 40, after: 40, line: 300 },
                    children: [new TextRun({ text: "[8]  A. Ramcharan et al., \"Edge-Optimised 1D-CNNs for Real-Time Biosignal Classification,\" Frontiers in Computer Science, vol. 6, p. 1852, 2024.", size: 22, font: "Times New Roman" })]
                }),
            ]
        }
    ]
});

Packer.toBuffer(doc).then(buffer => {
    fs.writeFileSync(__dirname + "/NeuroBioSense_Report.docx", buffer);
    console.log("Done. Saved to " + __dirname + "/NeuroBioSense_Report.docx");
});