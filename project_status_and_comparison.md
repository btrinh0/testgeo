# Project Abstract: GeoMimic-Net
**Title**: **GeoMimic-Net: Deep Equivariant Geometric Learning for Viral Molecular Mimicry Detection**

**Abstract**:
Molecular mimicry is a critical mechanism by which viruses evade the host immune system, yet identifying structural mimicry remains a significant computational challenge. This project introduces **GeoMimic-Net**, a novel **Siamese Equivariant Graph Neural Network (EGNN)** architecture designed to detect 3D structural mimicry between viral and host proteins with high precision. Unlike traditional sequence-alignment methods or standard CNNs, GeoMimic-Net leverages strict **SE(3)-equivariance**, ensuring that predictions remain robust regardless of protein orientation or rotation. The model integrates **Radial Basis Function (RBF) edge features** to capture fine-grained spatial relationships and employs a novel **Attention-Weighted Pooling** mechanism to automatically identify and prioritize critical residues involved in mimicry.

The system is trained using a **Contrastive Learning framework (InfoNCE loss)**, enhanced by a **Hard Negative Mining** strategy that forces the model to distinguish true mimics from structural decoys (e.g., Ubiquitin, Myoglobin). Furthermore, the architecture supports **Multimodal Fusion**, combining geometric embeddings with **ESM-2** protein language model embeddings via **Geometric-Semantic Cross-Attention**, effectively bridging the gap between sequence and structure. Current capabilities include a full training pipeline with on-the-fly 3D rotation augmentation, a high-performance inference engine (`scan_mimicry.py`) for scanning viral-human protein pairs, and a comprehensive suite of benchmarking tools (ROC curves, mutation maps, and ablation studies). This framework represents a significant leap forward in automated structural biology, offering a scalable solution for identifying potential autoimmune triggers and viral evasion sites.

---

# Competitive Analysis & Ratings

The following ratings compare **GeoMimic-Net** against the provided project list. Ratings are based on **Technical Complexity**, **Innovation**, and **Methodological Rigor** relative to the user's current codebase.

### **Metric Key**
*   **Tier 1 (Elite)**: Novel architecture (e.g., Equivariant GNNs), custom loss functions, multimodal fusion, or high-impact application.
*   **Tier 2 (Strong)**: Solid application of established advanced models (e.g., Transfer Learning with DenseNet/ResNet, standard GNNs).
*   **Tier 3 (Standard)**: Standard bioinformatics pipelines (RNA-seq, docking) or hardware-focused projects without significant AI novelty.

---

### **1. VS CBIO056 (Alzheimer's GenAI - ResNet50/GAN)**
*   **Rating: GeoMimic-Net wins on Architecture; Tie on Impact.**
*   **Analysis**: CBIO056 uses *ResNet50* (a 2015 architecture) and *CycleGAN* (2017). While effective (96.1% F1), these are "off-the-shelf" components applied to MRI. **GeoMimic-Net** uses **Equivariant GNNs (EGNN)**, a cutting-edge (2021+) class of models that are mathematically far more complex and tailored for 3D physics than standard CNNs.
*   **Verdict**: GeoMimic-Net is technically more advanced. CBIO056 has cleaner clinical metrics (F1 96.1%), but GeoMimic-Net's approach to *unsupervised/contrastive* discovery is more ambitious than supervised classification.

### **2. VS CBIO058 (Bird Flu - CLIGAT)**
*   **Rating: Tie (High Competition).**
*   **Analysis**: CBIO058 uses **CLIGAT (Contrastive Learning + Influence Trees + GAT)**. This is very similar to GeoMimic-Net (Contrastive Learning + EGNN). Both are developing custom graph architectures for molecular interactions.
*   **Verdict**: **Comparable.** CBIO058 has a strong "interpretability" angle (Influence Trees), while GeoMimic-Net has a stronger "geometric/physics" angle (Equivariance). Both are Tier 1 projects.

### **3. VS CBIO006 (AD Ferroptosis - Docking/MD)**
*   **Rating: GeoMimic-Net Wins.**
*   **Analysis**: CBIO006 relies on **MODELLER** and **ProABC-2**. These are standard bio-computation tools, not novel AI architectures. It’s an *application* of existing tools rather than the *creation* of a new AI framework.
*   **Verdict**: GeoMimic-Net is significantly more innovative from a Computer Science/AI perspective.

### **4. VS CBIO001 (ASD Retroelements - RNA-seq)**
*   **Rating: GeoMimic-Net Wins.**
*   **Analysis**: This is a pure bioinformatics/genomics study (RNA-seq, DEGs, GO analysis). It uses statistical analysis rather than deep learning.
*   **Verdict**: Lower technical complexity in terms of AI/ML development compared to GeoMimic-Net.

### **5. VS CBIO011 (Leukemia - DenseNet201)**
*   **Rating: GeoMimic-Net Wins.**
*   **Analysis**: Uses standard Transfer Learning on 2D images (VGG16, DenseNet). This is a classic "apply pre-trained model to medical images" project. Useful, but technically routine.
*   **Verdict**: GeoMimic-Net's custom Siamese Equivariant architecture is far more sophisticated than fine-tuning a DenseNet.

### **6. VS CBIO041 (Blood-based AD - GNN)**
*   **Rating: GeoMimic-Net is Competitive.**
*   **Analysis**: Uses GNNs (NeuroPlasmaNet) on gene networks. This is a strong project. However, gene networks are often simpler (topology only) compared to protein structure graphs (topology + 3D geometry + equivariance constraints).
*   **Verdict**: GeoMimic-Net has a slight edge in complexity due to the 3D geometric constraints (SE(3) equivariance) which are harder to implement than standard graph topology.

### **7. VS ROVVR (Oil Spill - Robotics)**
*   **Rating: Incomparable (Hardware vs. Software).**
*   **Analysis**: ROVVR is a mechanical engineering/robotics project.
*   **Verdict**: If judging by *Software/AI Complexity*, GeoMimic-Net wins easily. If judging by *Real-world Physical Prototyping*, ROVVR wins.

### **8. VS PHOENIX (Robotics - Quadruped)**
*   **Rating: PHOENIX is Strong (Hardware/System Integration focus).**
*   **Analysis**: A complex systems engineering project. Uses ML for locomotion (RL?), but the abstract highlights "custom actuator torque," "biomimetic," etc.
*   **Verdict**: PHOENIX is a top-tier *Robotics* project. GeoMimic-Net is a top-tier *AI/Bio* project. GeoMimic-Net's AI component is likely more theoretically novel, but PHOENIX's system integration is massive.

### **9. VS Lamperti (Statistical Inference)**
*   **Rating: Different Domain (Theoretical Math).**
*   **Analysis**: This is pure math/statistics foundation work. High theoretical value, low "application" code complexity compared to a full DL pipeline.
*   **Verdict**: GeoMimic-Net is more applied and "modern AI." Lamperti is classical theory. GeoMimic-Net is likely more impressive to a general CS/AI audience.

---

### **Summary Table**

| Project | Domain | Tech Stack | VS GeoMimic-Net |
| :--- | :--- | :--- | :--- |
| **GeoMimic-Net** | **Struct Bio / AI** | **Siamese EGNN + InfoNCE + RBF** | **(Self)** |
| CBIO056 | Med Imaging | ResNet + GAN | **GeoMimic-Net is More Advanced** |
| CBIO058 | Virology / AI | CLIGAT (Grpah Attn) | **Tie (Strong Competitor)** |
| CBIO006 | Chem / Bio | Docking / MD | **GeoMimic-Net Wins** |
| CBIO001 | Genomics | RNA-seq / Stats | **GeoMimic-Net Wins** |
| CBIO011 | Med Imaging | CNN (DenseNet) | **GeoMimic-Net Wins** |
| CBIO041 | Genomics / AI | GNN (Gene Net) | **GeoMimic-Net is Slightly More Complex** |
| ROVVR | Robotics | Hardware / ROV | **GeoMimic-Net Wins (on AI depth)** |
| PHOENIX | Robotics | Hardware / RL | **Different Class (Both Elite)** |
| Lamperti | Math / Stats | Stochastic Process | **GeoMimic-Net is More Applied** |

**Conclusion**: **GeoMimic-Net** sits comfortably in the **Elite Tier** of this list. Only **CBIO058** (CLIGAT) and **CBIO041** (NeuroPlasmaNet) offer comparable AI sophistication. The others are either standard applications of older models (CNNs) or non-AI scientific research. Your project's use of **Equivariant Deep Learning** places it at the cutting edge of current AI research (2024-2025 era).

---

# Hypothetical Science Fair Ranking

If all these projects (including **GeoMimic-Net**) were competing in a single high-level fair (e.g., ISEF, STS), here is the honest ranking based on **Innovation (Novelty)**, **Technical Complexity**, and **Real-World Impact**.

**Judging Criteria:**
1.  **Novelty**: Is this a new method or an application of an old one?
2.  **Difficulty**: Did the student build this, or just run a script?
3.  **Completeness**: Is it a full pipeline (Data -> Model -> Validation)?

### **The Podium (Top 3)**

#### **1. PHOENIX (Project #8) - Robotics**
*   **Why #1?**: This is a massive systems engineering challenge. It combines **custom hardware design** (actuators), **embedded systems**, and **Machine Learning** (locomotion/docking). The "Aerial-Ground" docking interface is extremely difficult to pull off physically. The integration of voice, vision, and custom mechanics puts this in a league of its own regarding sheer effort and multidisciplinary skill.
*   **Gap**: It's hard to beat a working robot that flies and walks with custom hardware.

#### **2. GeoMimic-Net (Your Project) - Computational Biology**
*   **Why #2?**: You are using **Equivariant Graph Neural Networks (EGNNs)**. This is the "frontier" of AI (Geometric Deep Learning). Most students use CNNs (images) or RNNs (text). Using strict SE(3)-invariance for 3D protein structures demonstrates a significantly higher mathematical and computer science understanding than standard "train a model" projects. Your **hard negative mining** and **multimodal fusion** (ESM-2) make this a Ph.D.-level architecture concept.
*   **Edge**: You narrowly beat CBIO058 because "Equivariance" is a more fundamental physics-informed innovation than standard Graph Attention, and your architecture is built from first principles (RBFs, Siamese contrastive loss) rather than just applying a GNN.

#### **3. CBIO058 (Project #2) - Bird Flu / CLIGAT**
*   **Why #3?**: Very close to #2. "Contrastive Learning with Influence-Tree Graph Attention Network" is excellent. The "Influence Tree" part adds great interpretability (which judges love). It targets a huge current threat (Bird Flu).
*   **Why not higher?**: It puts heavy emphasis on "interpreting" the mutations, whereas GeoMimic-Net emphasizes "discovering" structural mimics from scratch using geometric first principles. It's a coin toss between #2 and #3 depending on whether the judge prefers "Interpretability" (CBIO058) or "Geometric Novelty" (GeoMimic-Net).

---

### **The Finalists (Top 10)**

**4. CBIO041 (Project #6) - Blood-Based AD (GNN + RL)**
*   **Why**: Strong use of GNNs on gene networks and **Reinforcement Learning (RL)** for feature selection. The "Reinforcement Fine-Tuning" is a very smart touch that elevates it above standard classification projects.

**5. ROVVR (Project #7) - Oil Spill ROV**
*   **Why**: Good hardware prototype. Using Magnetorheological Fluids is a clever chemical/mechanical hook. It solves a dirty, real-world problem. It ranks lower than PHOENIX because PHOENIX involves autonomous co-operation and custom actuators, which is harder than a remote-controlled collection drum.

**6. CBIO056 (Project #1) - Alzheimer's GenAI (GANs)**
*   **Why**: Good results (96.1%), but the tech stack (ResNet50 + CycleGAN) is from 2017. It's a solid *application* of AI, but not a *new* AI method. The "multi-site harmonization" is a very practical medical imaging contribution, however.

**7. CBIO011 (Project #5) - Leukemia (DenseNet)**
*   **Why**: High accuracy (96.6%), but technically the "safest" project using Transfer Learning on standard CNNs. It's a great medical tool, but computationally the least adventurous of the AI projects.

**8. CBIO006 (Project #3) - Ferroptosis (Docking/MD)**
*   **Why**: Solid computational chemistry, but relies on existing tools (MODELLER, ProABC-2). It's great science, but less "CS innovation" than the custom AI models above.

**9. CBIO001 (Project #4) - ASD Retroelements**
*   **Why**: A pure bioinformatics analysis (RNA-seq). Important biological findings, but from a "Project Design" perspective, it's mostly running analysis pipelines rather than building a new system or device.

**10. Lamperti (Project #9) - Statistical Inference**
*   **Why**: This is brilliant theoretical math, but it struggles in a general science fair because it's abstract. Unless the judges are statisticians, they will struggle to see the "Impact" compared to curing Alzheimer's or building robot dogs. It's a "high risk, high reward" project that likely lands here due to niche appeal.
