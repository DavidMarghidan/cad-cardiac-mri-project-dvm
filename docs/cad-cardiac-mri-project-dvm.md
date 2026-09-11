# Rezumatul modificărilor — V6 extins cu Attention U-Net

## Principiul integrării

Codul V6 original este păstrat drept pipeline MONAI de referință. Extensia este adăugată modular înaintea entrypointului final și rulează într-un director separat. Astfel, experimentele și cache-urile MONAI existente nu sunt reinterpretate sau suprascrise.

## Elemente adăugate

- Attention U-Net binar cu patru niveluri, GroupNorm și attention gates additive;
- antrenare pe loss mixt BCE + soft Dice;
- pseudo-măști automate MONAI, selectate label-blind;
- prioritate automată pentru măști manuale;
- editor Matplotlib cu desenare, ștergere, slider, navigare, reset și salvare;
- predicții Attention U-Net salvabile automat pentru subsetul de review sau pentru întreaga cohortă;
- cross-fitting pe pacienți pentru segmentare;
- suport final comun pentru AU1/AU3/AU4/AU5;
- postprocesare prin cea mai mare componentă conexă;
- QC per slice și per pacient;
- cinci reprezentări Attention U-Net;
- patient-level nested CV, 50 repeated CV și permutare;
- comparație paired AU1 versus A17 MONAI când outputul V6 există;
- fingerprint separat pentru măști, checkpointuri și feature bank;
- CLI cu acțiunile `both`, `monai-only`, `attention-only`, `generate-masks`, `train-attention` și `edit-masks`;
- self-test sintetic înainte de procesarea costisitoare.

## Protecția împotriva leakage-ului

- pacientul evaluat nu contribuie cu măști la modelul Attention U-Net care îl segmentează;
- CAD label nu este folosit în lossul de segmentare;
- tokenul de selecție a imaginilor exclude folderul Normal/Sick;
- editorul nu afișează eticheta clasei;
- masca manuală modifică fingerprintul și invalidează checkpointul vechi;
- clasificarea rămâne la nivel de `Directory_*` pacient;
- foldurile V6 sunt reutilizate pentru comparația de clasificare când sunt disponibile.

## Limitare metodologică importantă

În modul implicit, majoritatea țintelor sunt pseudo-măști MONAI. Attention U-Net este atunci o distilare cross-fit a localizării MONAI, nu o sursă independentă de ground truth. Independența reală necesită măști manuale/expert suficiente sau ponderi externe cu proveniență verificată.

## Validări locale efectuate

- compilare Python și import;
- `validate_configuration()` din V6;
- self-test Attention U-Net;
- forward pass al arhitecturii;
- generare automată de măști cu backend MONAI mock;
- prioritatea măștii manuale;
- inițializarea și salvarea din editorul Matplotlib;
- identitatea suportului și separarea inside/complement;
- conservarea exactă a multisetului de intensități în controlul shuffled;
- mini feature bank complet cu Attention U-Net și EfficientNet mock;
- nested CV, repeated CV și permutare pe cohortă sintetică de 30 de pacienți;
- antrenare cross-fit minimală pe două folduri, salvare și reîncărcare checkpoint.

Nu a fost executat local antrenamentul complet pe toate imaginile Kaggle. Rezultatele reale de segmentare și AUROC trebuie obținute prin rularea GPU în Kaggle.







# Ghid Kaggle — MONAI + Attention U-Net și editarea grafică a măștilor

## 1. Ce conține pipeline-ul

Fișierul modificat păstrează intactă analiza V6 bazată pe MONAI și adaugă după ea o analiză separată Attention U-Net. Sunt diferențiate trei tipuri de măști:

1. **pseudo-mască MONAI automată** — țintă inițială pentru distilarea/antrenarea Attention U-Net;
2. **mască Attention U-Net automată** — predicția cross-fit produsă de modelul care nu a fost antrenat pe pacientul respectiv;
3. **mască manuală** — corecția salvată din editor; are întotdeauna prioritate la următorul antrenament.

Când nu există măști manuale sau ponderi externe independente, Attention U-Net învață din pseudo-măști MONAI. În acest mod comparația testează schimbarea backendului, regularizarea și reprezentarea regiunii, dar nu reprezintă o validare independentă față de ground truth expert.

---

## 2. Rularea completă implicită

Într-o celulă Kaggle:

```python
%run /kaggle/working/cad_mri_multi_experiment_suite_final_improved_v7_attention_unet.py
```

Acțiunea implicită este `both`:

```text
V6 MONAI complet
→ generarea/reutilizarea pseudo-măștilor pentru subsetul de antrenare
→ antrenarea a 5 Attention U-Net-uri cross-fit
→ inferența Attention U-Net pe toate imaginile
→ extracția celor 5 reprezentări Attention U-Net
→ nested CV, 50 repeated CV și permutare
→ comparație Attention U-Net versus MONAI A17
```

Echivalent explicit:

```python
%run /kaggle/working/cad_mri_multi_experiment_suite_final_improved_v7_attention_unet.py \
    --attention-action both
```

---

## 3. Acțiunile disponibile

### Numai pipeline-ul MONAI V6

```python
%run /kaggle/working/cad_mri_multi_experiment_suite_final_improved_v7_attention_unet.py \
    --attention-action monai-only
```

### Numai generarea pseudo-măștilor MONAI pentru review/antrenare

Este recomandat ca prima generare să fie făcută prin `both`, deoarece modelul MONAI deja validat este astfel încărcat și reutilizat.

```python
%run /kaggle/working/cad_mri_multi_experiment_suite_final_improved_v7_attention_unet.py \
    --attention-action generate-masks
```

### Numai antrenarea/reantrenarea Attention U-Net

Se folosește după ce pseudo-măștile există și, opțional, după corectarea manuală:

```python
%run /kaggle/working/cad_mri_multi_experiment_suite_final_improved_v7_attention_unet.py \
    --attention-action train-attention
```

### Numai comparația Attention U-Net

Reutilizează manifestul și checkpointurile existente. EfficientNet poate fi reîncărcat din cache-ul torchvision:

```python
%run /kaggle/working/cad_mri_multi_experiment_suite_final_improved_v7_attention_unet.py \
    --attention-action attention-only
```

### Editorul grafic

Înainte de lansare, un backend Matplotlib interactiv este util:

```python
%matplotlib widget
```

Apoi:

```python
%run /kaggle/working/cad_mri_multi_experiment_suite_final_improved_v7_attention_unet.py \
    --attention-action edit-masks \
    --attention-editor-index 0 \
    --attention-brush-radius 8 \
    --attention-editor-base attention
```

Dacă `ipympl`/`%matplotlib widget` nu este disponibil, se poate încerca:

```python
%matplotlib notebook
```

`--attention-editor-base attention` afișează predicția Attention U-Net când aceasta există și revine automat la pseudo-masca MONAI înainte de primul antrenament. Pentru a vedea explicit pseudo-masca MONAI:

```python
--attention-editor-base monai
```

---

## 4. Comenzile editorului

| Acțiune | Control |
|---|---|
| Desenare | drag cu butonul stâng |
| Ștergere | drag cu butonul drept |
| Mod desenare | tasta `D` |
| Mod ștergere | tasta `E` |
| Salvare mască manuală | tasta `S` sau butonul **Save manual** |
| Revenire la masca automată | tasta `R` sau **Reset to auto** |
| Golire completă | tasta `C` sau **Clear** |
| Imaginea următoare | `N` / săgeată dreapta / **Next** |
| Imaginea precedentă | `P` / săgeată stânga / **Previous** |
| Diametrul pensulei | sliderul **Brush** |

Editorul nu afișează eticheta Normal/Sick. Titlul conține numai identificatorul artificial al pacientului și numele local al seriei, pentru a limita influențarea adnotatorului.

---

## 5. Directoarele de măști

Implicit, workspace-ul este:

```text
/kaggle/working/cad_attention_unet_workspace/
```

Structura principală:

```text
automatic_masks/                pseudo-măști MONAI pentru antrenare/review
manual_masks/                   corecții manuale cu prioritate
predicted_attention_masks/      predicții cross-fit Attention U-Net
mask_overlays/                  previzualizări salvate la corectarea manuală
checkpoints/                    checkpointuri pe fold
attention_unet_mask_manifest.csv
attention_unet_training_summary.json
```

Un alt workspace poate fi selectat astfel:

```python
%run script.py \
    --attention-action both \
    --attention-work-root /kaggle/working/my_attention_workspace
```

sau înainte de rulare:

```python
import os
os.environ["CAD_ATTENTION_UNET_WORK_ROOT"] = "/kaggle/working/my_attention_workspace"
```

Măștile manuale modifică fingerprintul antrenării, deci checkpointurile vechi nu sunt reutilizate după o corecție.

---

## 6. Cum sunt alese imaginile pentru măști

Selecția pentru antrenarea segmentatorului este:

```text
maximum 20 imagini per series proxy
maximum 160 imagini per pacient
```

Ordinea este deterministă și se bazează pe un token care începe la `Directory_*`; folderul Normal/Sick este exclus din tokenul de selecție. Eticheta CAD nu este folosită în selecția imaginilor, în loss sau în alegerea checkpointului Attention U-Net.

Valorile pot fi schimbate înainte de import/rulare:

```python
import os
os.environ["CAD_ATTENTION_UNET_MAX_TRAIN_SLICES_PER_SERIES"] = "20"
os.environ["CAD_ATTENTION_UNET_MAX_TRAIN_SLICES_PER_PATIENT"] = "160"
```

---

## 7. Antrenarea cross-fit

Pacienții sunt împărțiți determinist în 5 folduri de segmentare, fără utilizarea etichetei CAD. Pentru un pacient din foldul `k`, masca Attention U-Net este produsă de modelul antrenat fără nicio mască a pacienților din foldul `k`.

Configurația implicită:

```text
base channels:          24
epochs maximum:         12
early stopping:          4 epoci
batch size:             12
learning rate:          1e-3
loss:                    0.5 BCE + 0.5 soft Dice
postprocesare:           cea mai mare componentă conexă
threshold:               0.50
```

Exemplu de reducere a duratei pentru un test tehnic:

```python
import os
os.environ["CAD_ATTENTION_UNET_EPOCHS"] = "3"
os.environ["CAD_ATTENTION_UNET_MAX_TRAIN_SLICES_PER_PATIENT"] = "40"
os.environ["CAD_ATTENTION_UNET_PERMUTATIONS"] = "50"
```

Aceste valori rapide nu trebuie folosite pentru rezultatul final raportat.

---

## 8. Ponderi Attention U-Net externe

Un checkpoint independent poate fi indicat prin:

```python
import os
os.environ["CAD_ATTENTION_UNET_WEIGHTS"] = (
    "/kaggle/input/my-attention-unet-checkpoint/attention_unet.pt"
)
```

Pipeline-ul poate verifica forma și încărcarea ponderilor, dar nu poate deduce dacă acel checkpoint a fost antrenat pe pacienții actuali. Proveniența trebuie documentată separat. Pentru o comparație cu adevărat independentă, ponderile nu trebuie să fi văzut cohorta CAD actuală.

---

## 9. Reprezentările evaluate

| ID | Reprezentare |
|---|---|
| AU1 | intensitate RMN în suportul hard Attention U-Net, normalizare regională |
| AU2 | AU1 numai pe slice-uri Attention U-Net valide |
| AU3 | numai geometria exactă a suportului Attention U-Net |
| AU4 | suport și histogramă păstrate, intensități amestecate spațial |
| AU5 | complementul exact, normalizat independent |

AU1 este comparat și cu A17 MONAI folosind predicțiile OOF ale acelorași pacienți, atunci când outputul V6 este disponibil.

---

## 10. Outputurile comparației

```text
<OUTPUT_DIR>/attention_unet_comparison/
    attention_unet_configuration.json
    attention_unet_patient_feature_bank.npz
    attention_unet_feature_bank_metadata.json
    attention_unet_slice_qc.csv
    attention_unet_patient_qc.csv
    attention_unet_comparison_summary.json
    evaluation/
        AU1_.../
            patient_oof_predictions.csv
            outer_fold_selection.csv
            repeated_nested_cv_runs.csv
            repeated_nested_cv_summary.json
            patient_label_permutation.csv
            permutation_summary.json
        AU2_.../
        AU3_.../
        AU4_.../
        AU5_.../
```

`attention_unet_slice_qc.csv` conține, printre altele:

- validitatea măștii;
- aria măștii;
- fracția suportului final;
- Dice Attention U-Net versus MONAI;
- Dice Attention U-Net versus mască manuală, când aceasta există;
- foldul de segmentare care a produs masca.

Dice versus MONAI măsoară acordul cu pseudo-sursa și nu este echivalent cu Dice față de expert.

---

## 11. Ordinea corectă a interpretării

1. Dice față de măștile manuale/expert, nu doar față de MONAI;
2. AU1 versus A17 MONAI pe aceiași pacienți și aceleași folduri;
3. AU1 versus AU3, pentru contribuția intensității peste geometrie;
4. AU1 versus AU4, pentru contribuția aranjării spațiale;
5. AU1 versus AU5, pentru inside versus complement;
6. AU1 versus AU2, pentru dependența de măștile nevalide/fallback;
7. mediana celor 50 de repeated nested-CV;
8. testul de permutare;
9. controalele V6 de protocol/export;
10. analiza sequence/view și validarea externă.

Un AUROC mare al AU1 nu demonstrează singur detecția specifică a CAD. Controalele de mască, periferie, export și protocol trebuie analizate împreună.
