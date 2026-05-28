# Workflow-CI

Repository ini dibuat untuk memenuhi **Kriteria 3: Membuat Workflow CI**.

## Struktur Folder

```text
Workflow-CI
├── .github/workflows/ci.yml
├── .workflow/ci.yml
└── MLProject
    ├── modelling.py
    ├── conda.yaml
    ├── MLProject
    ├── requirements.txt
    ├── DockerHub.txt
    └── breast_cancer_preprocessing
        ├── train_preprocessed.csv
        ├── test_preprocessed.csv
        └── breast_cancer_preprocessed.csv
```

## Cara Menjalankan Lokal

Masuk ke folder `Workflow-CI`, lalu jalankan:

```bash
pip install -r MLProject/requirements.txt
mlflow run MLProject --env-manager=local
```

Hasil training akan masuk ke:

```text
MLProject/mlruns
MLProject/outputs
```

## GitHub Actions

Workflow berada di:

```text
.github/workflows/ci.yml
```

Trigger:

- `push` ke branch `main` atau `master`
- `pull_request` ke branch `main` atau `master`
- manual melalui `workflow_dispatch`

