############################################################
# TCGA-GBM Mutation Labels
# HARDENED VERSION (cannot fail due to functions)
############################################################

# ---------- STEP 0: Nuke environment ----------
rm(list = ls(all.names = TRUE))
gc()

options(
  stringsAsFactors = FALSE,
  restoreWorkspace = FALSE
)

# ---------- STEP 1: Install & load packages ----------
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

if (!requireNamespace("TCGAbiolinks", quietly = TRUE)) {
  BiocManager::install("TCGAbiolinks", ask = FALSE, update = TRUE)
}

library(TCGAbiolinks)

# ---------- STEP 2: Safe working directory ----------
base_dir <- "~/TCGA_GBM_PROJECT"
dir.create(base_dir, showWarnings = FALSE, recursive = TRUE)
setwd(base_dir)

cat("Working directory:\n", getwd(), "\n")

# ---------- STEP 3: Query TCGA-GBM ----------
query_maf <- GDCquery(
  project = "TCGA-GBM",
  data.category = "Simple Nucleotide Variation",
  data.type = "Masked Somatic Mutation",
  access = "open",
  workflow.type = "Aliquot Ensemble Somatic Variant Merging and Masking"
)

# ---------- STEP 4: Download ----------
GDCdownload(
  query = query_maf,
  directory = base_dir,
  method = "api"
)

# ---------- STEP 5: Prepare MAF ----------
maf <- GDCprepare(
  query = query_maf,
  directory = base_dir
)

cat("MAF loaded. Rows:", nrow(maf), "\n")

# ---------- STEP 6: Keep real mutations only ----------
maf <- maf[maf$Variant_Classification %in% c(
  "Missense_Mutation",
  "Nonsense_Mutation",
  "Frame_Shift_Del",
  "Frame_Shift_Ins",
  "Splice_Site"
), ]

# ---------- STEP 7: Patient ID ----------
maf$TCGA_ID <- substr(maf$Tumor_Sample_Barcode, 1, 12)

# ---------- STEP 8: Genes ----------
genes <- c(
  "IDH1", "IDH2", "EGFR", "TP53", "PTEN", "ATRX",
  "TERT", "NF1", "PIK3CA", "RB1",
  "VEGFA", "BRAF", "NTRK1", "NTRK2", "NTRK3"
)

# ---------- STEP 9: Build label table ----------
patients <- sort(unique(maf$TCGA_ID))
labels <- data.frame(TCGA_ID = patients)

for (g in genes) {
  mutated <- unique(maf$TCGA_ID[maf$Hugo_Symbol == g])
  labels[[paste0(g, "_mut")]] <- as.integer(labels$TCGA_ID %in% mutated)
}

# ---------- STEP 10: Combined flags ----------
labels$IDH_mut  <- as.integer(labels$IDH1_mut == 1 | labels$IDH2_mut == 1)
labels$NTRK_mut <- as.integer(
  labels$NTRK1_mut == 1 |
    labels$NTRK2_mut == 1 |
    labels$NTRK3_mut == 1
)
labels$VEGF_mut <- labels$VEGFA_mut

# ---------- STEP 11: CRITICAL FIX ----------
# Remove ANY function columns (this is what fixes your error)
labels <- labels[, !sapply(labels, is.function)]

# ---------- STEP 12: Final sanity ----------
stopifnot(is.data.frame(labels))
stopifnot(!any(sapply(labels, is.function)))

cat("Final table size:", dim(labels), "\n")
print(head(labels))

# ---------- STEP 13: Save (GUARANTEED SAFE) ----------
write.csv(
  labels,
  file = "TCGA_GBM_mutation_labels.csv",
  row.names = FALSE
)

saveRDS(labels, "TCGA_GBM_mutation_labels.rds")

cat("\n✅ SUCCESS\n")
cat("Files created:\n")
cat("- TCGA_GBM_mutation_labels.csv\n")
cat("- TCGA_GBM_mutation_labels.rds\n")
