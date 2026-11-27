# BevNetDeep – tanítás és járműdetektálás

[English](README.md) · **Magyar**

Ha saját légi lézeres felmérésen keresel járműveket, a **detektálási példával** kezdj. Ehhez a repóban már megvan a betanított modell. A **tanítási példa** egy kisebb hálózatot tanít be generált adatokon, majd visszatölti a mentett súlyokat. Mindkét példa futtatható CPU-n.

## Telepítés

Nyiss egy terminált a repó gyökerében, ahol a `pyproject.toml` és a `models/` könyvtár található. Aktiválj egy Python-környezetet (a példákat Python 3.12-vel teszteltük), majd futtasd:

```text
python -m pip install .
git lfs pull
```

Az első parancs telepíti a Python-csomagot és a `bordereye-bev-*` parancsokat. A második a Git LFS-sel kezelt két súlyfájlt tölti le. A további parancsokat ugyanebből a könyvtárból és környezetből futtasd.

## Járműdetektálás saját LAS/LAZ pontfelhőn

A `--input` után add meg egy méterben megadott koordinátájú pontfelhő útvonalát. Térképi megjelenítéshez legyen helyes a fájl vetületi információja. A példában a bemenet neve `parking.laz`:

```text
bordereye-bev-detect --input "D:/LiDAR/parking.laz" --weights models/bevnetdeep-vehicle-detection.pt --output-dir output/parking --device cpu --no-amp --confidence 0.25 --decode-score-mode hm_seg_center_obb --nms-iou 0.20
```

A program alapbeállítással a LAS 2-es talajosztályát használja, ha van elegendő talajpont és legalább egy felszín feletti pont. Egyébként morfológiai talajszűrést végez. A talajhoz viszonyított magasságokból és a pontok térbeli eloszlásából elkészíti a modell 12 felülnézeti (BEV) csatornáját. A `--confidence` a detekciók minimális pontszáma. A `--nms-iou` az egymást átfedő, elforgatott dobozok szűrését vezérli.

| Fájl az `output/parking/` könyvtárban | Mire használható? |
|---|---|
| `parking_detections.png` | Első vizuális ellenőrzés. Láthatod, hová kerültek a detektált járművek dobozai. |
| `parking_boxes.csv` | Detekciónként egy sor. A `cx`, `cy` a középpont, `L` és `W` a hossz és a szélesség méterben, `theta_deg` az elfordulás fokban, `score` a detekció pontszáma. |
| `parking_vehicles.geojson` | A járműdobozok térképi ellenőrzéséhez például QGIS-ben. |
| `parking_detections.json` | Gépi feldolgozáshoz. Tartalmazza a pontok és detekciók számát, az alkalmazott talajosztályozást és az eredményfájlok útvonalát. |
| `parking_mask.ers` és a hozzá tartozó bináris raszter | A szegmentálásból kapott járműmaszk. A két fájlt együtt tartsd. |

Saját felmérés nélkül a következő parancs egy szintetikus parkolóban futtatja ugyanezt a betanított modellt, és a terminálban kiírja a találatok számát:

```text
python examples/synthetic_inference.py --weights models/bevnetdeep-vehicle-detection.pt
```

## Tanítási példa CPU-n, saját adatok nélkül

**1. Készíts kis tanítóadat-készletet.** A generátor négy címkézett jelenetet ír az `output/tutorial/data/train/` és egy külön tesztjelenetet az `output/tutorial/data/test/` könyvtárba. A csempék `.npz` fájlok.

```text
python -m bordereye.ml.bev.data.build_dataset --sources synth --out output/tutorial/data --n-train 4 --n-test 1 --tile-size 32 --veh-per-scene 2 4 --workers 1 --no-viz
```

**2. Taníts egy kis modellt.** A konfiguráció egy epochot, kétmintás kötegeket és 64 × 64 pixeles kivágásokat állít be. A négy tanításra generált csempéből három kerül tanításra, egy validálásra. A külön tesztcsempe későbbi kiértékeléshez használható. [A példa beállításai](training-cpu.json)

```text
bordereye-bev-train --config examples/training-cpu.json --dataset-dir output/tutorial/data --output-dir output/tutorial/training
```

**3. Töltsd vissza a saját súlyokat és futtass predikciót.**

```text
python examples/synthetic_inference.py --weights output/tutorial/training/tutorial.pt
```

Az `output/tutorial/training/tutorial.pt` a validáció alapján kiválasztott modell, a `tutorial_last.pt` az utolsó epoch mentése. A `train_log.csv` az epochonkénti mutatókat, a `training_curves.png` a tanulási görbéket tartalmazza. Az egyepochos példa a teljes munkamenet kipróbálására szolgál. Saját felmérés kiértékeléséhez a fenti detektálási parancsban szereplő, kutatási adatokon betanított modellt használd.

## Hol találom a cikk kísérletéhez szükséges fájlokat?

| Útvonal a repó gyökeréhez képest | Szerepe |
|---|---|
| `models/bevnetdeep-vehicle-detection.pt` | A cikk részletes kiértékelésében használt detektor. Ezt add meg inferenciához. |
| `models/bevnetdeep-training-warm-start.pt` | A cikk tanítási futásának kezdősúlyai. Ezt add meg a `--resume` kapcsolóval a kutatási konfigurációhoz. |
| `configs/seed-123.json` | A teljes tanítás beállításai, köztük a CUDA-eszköz, a tanulási ráta és az adatforrások súlyozása. |
| `splits/seed-123-train-val.json` | A rögzített tanító- és validációs csempék fájlnevei. |
| `splits/test-tiles.json`, `results/` | Tesztcsempék listája és a közölt mérés archivált eredménytáblái. |
| `examples/synthetic_inference.py` | Python-példa a BEV-csatornák előállítására, a modell betöltésére és a `BevPredictor.predict()` hívására. |

A teljes tanításhoz a szintetikus, ritkított szintetikus, DALES és WiSAR3D csempéket tartalmazó, **előkészített** adatkönyvtárat add meg. A `--dataset-dir` a `train/` és `test/` szülőkönyvtára. A letölthető nyers adatok mellé az eredeti kísérletben használt származtatott járműdobozok, figyelmen kívül hagyott dobozok és WiSAR3D-mozaikok is szükségesek. [Adatletöltések és reprodukálhatóság](../README.md#data-and-evaluation)

```text
bordereye-bev-train --config configs/seed-123.json --dataset-dir "D:/LiDAR/prepared-bev" --split-manifest splits/seed-123-train-val.json --resume models/bevnetdeep-training-warm-start.pt --output-dir output/seed-123
```

Ez a recept a kezdősúlyokból, friss optimalizálóval indul. Az `output/seed-123/best_model_consistency.pt` fájlt ugyanúgy átadhatod a detektálási parancs `--weights` kapcsolójának, mint a repóval szállított modellt.
