# Revizuire pipeline CAD Cardiac MRI — 22 august 2026

## Invariant păstrat

`patient_id` rămâne **exact numele folderului `Directory_*`**. Codul nu derivă pacientul din `SR_*`, din folderele `series*`, din numele imaginilor sau din grupurile de duplicate. Folderele copil imediate sunt tratate doar drept **proxy-uri operaționale de serie**, deoarece exportul JPEG nu oferă un `SeriesInstanceUID` DICOM verificabil.

## Corecții metodologice și de comentarii

1. Pipeline-ul este descris acum drept un flux complet de execuție, nu drept o rețea neurală „end-to-end” antrenată simultan: segmentatorul MONAI și EfficientNet sunt înghețate.
2. Numărul efectiv de observații etichetate este numărul de foldere `Directory_*` descoperite, nu numărul de imagini și nici numărul de 1.224 participanți raportat de articol.
3. Ieșirile `predict_proba` și valorile obținute prin fuziune sunt denumite scoruri necalibrate; nu sunt prezentate ca probabilități clinice posterioare validate.
4. Folderele copil sunt denumite „series proxies”, nu serii DICOM validate.
5. Preprocesarea EfficientNet este descrisă corect drept o adaptare: se păstrează întregul canvas aliniat 256×256 și se redimensionează la 224×224, în locul transformării torchvision complete cu resize și center crop.
6. Comentariul despre ponderile globale a fost corectat: în Logistic Regression regularizată, multiplicarea tuturor ponderilor cu o constantă schimbă balanța dintre loss și regularizare. Pentru metoda legacy, masa totală a ponderilor este acum numărul pacienților de antrenare, nu numărul slice-urilor.
7. Limitarea MONAI este formulată explicit: modelul este destinat imaginilor RM cardiac short-axis; gate-ul și fallback-ul nu demonstrează corectitudinea anatomică pe toate secvențele CAD.

## Îmbunătățiri implementate

- **Strategie implicită la nivel de pacient:** embedding-urile slice-urilor sunt agregate mai întâi în fiecare proxy de serie și apoi egal între proxy-urile pacientului. Clasificatorul vede un singur vector pentru fiecare `Directory_*`.
- Metoda veche „clasificator pe slice + fuziune log-odds” rămâne disponibilă doar ca ablație prin `CLASSIFICATION_STRATEGY = "slice_probability_fusion"`.
- PCA opțională este ajustată exclusiv în foldul de antrenare și este folosită implicit pentru raportul foarte mare dintre 1.280 de caracteristici și numărul mic de pacienți.
- Împărțirea este `StratifiedKFold` exclusiv pe tabelul de pacienți; fiecare pacient primește exact un scor out-of-fold.
- AUC-ul principal este pooled OOF la nivel de pacient, cu interval bootstrap stratificat la nivel de pacient.
- Audit automat al duplicatelor exacte pe matricea grayscale decodificată. Sunt raportate separat duplicatele între pacienți și duplicatele între etichete; identitatea pacientului nu este modificată.
- QC MONAI separat pe clasă și pe pacient: rată gate valid, area ratio, peak probability și mean foreground probability. Acestea sunt diagnostice, nu scoruri de acuratețe a segmentării.
- Cache complet pentru embedding-urile înghețate, cu fingerprint care include datasetul, revizia modelului, setările ROI, versiunile software și configurația de inferență.
- Fișiere de ieșire distincte pentru fiecare configurație/ablație, pentru a preveni suprascrierea rezultatelor.
- Salvare CSV/JSON pentru predicții OOF, folduri, QC, duplicate și metadatele rularii.
- Încărcare MONAI optimizată: traseul normal descarcă artefactul oficial `models/model.ts` de la o revizie Hugging Face fixă, verifică SHA-256 și îl încarcă direct cu `torch.jit.load`. Importul lent `monai.networks.nets.UNet` apare numai în fallback-ul excepțional din `model.pt`.
- Fallback-ul verifică `train.json`, încarcă `model.pt` cu `weights_only=True`, folosește `strict=True`, validează outputul și abia apoi creează un cache TorchScript local.
- Modelul EfficientNet este fixat explicit la `IMAGENET1K_V1`, nu la aliasul schimbabil `DEFAULT`.
- Inference AMP opțională pe CUDA, selecție QC deterministă și `main()` protejat pentru compatibilitate cu DataLoader multiprocessing pe Windows.

## Validare efectuată în mediul de lucru

- compilare Python cu `python -m py_compile`;
- import al modulului și validarea configurației;
- încărcarea și validarea traseului oficial `model.ts` cu un artefact TorchScript sintetic local;
- validarea configurației fallback `train.json`;
- extracția de caracteristici pe imagini sintetice, atât cu MONAI ROI, cât și în ablația full-image;
- salvare și reîncărcare a feature cache-ului;
- detectarea unui grup sintetic duplicat între doi pacienți și două etichete;
- agregarea la un vector per pacient și verificarea masei ponderilor;
- câte un ciclu complet 5-fold pentru ambele strategii de clasificare.

Nu a fost executată o rulare completă pe toate imaginile datasetului real în acest mediu, deoarece datasetul și checkpointurile nu sunt montate aici. Descărcarea efectivă din Hugging Face nu a putut fi testată în containerul fără acces DNS; logica de descărcare folosește API-ul oficial `huggingface_hub`, iar traseul local rezultat a fost testat sintetic.

## Limitări care rămân

- `Directory_*` produce un cohort computațional foarte mic; intervalele de incertitudine și variația între folduri vor fi mari.
- Eticheta este la nivel de pacient, iar nu toate imaginile sunt neapărat individual informative pentru CAD.
- Proxy-urile de serie nu înlocuiesc metadatele DICOM originale.
- Segmentarea nu are ground truth în acest dataset; gate-ul MONAI necesită audit vizual și, ideal, validare manuală pe un subset.
- Duplicatele aproape identice/re-encodate nu sunt detectate de hash-ul exact; este necesar și un audit perceptual separat.
- Rezultatele interne nu înlocuiesc o validare externă pe date independente de spital.
