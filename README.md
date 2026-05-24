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

## Target Penilaian

### Basic 2 pts

Terpenuhi karena repository memiliki folder `MLProject` dan workflow CI yang menjalankan training model melalui MLflow Project.

### Skilled 3 pts

Terpenuhi karena workflow menyimpan hasil training sebagai GitHub Actions artifact:

- `MLProject/mlruns`
- `MLProject/outputs`

### Advance 4 pts

Workflow sudah disiapkan untuk build dan push Docker image menggunakan:

```bash
mlflow models build-docker
```

Agar bagian Advance berjalan, tambahkan secret GitHub:

- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`

Lalu isi link Docker Hub pada file `MLProject/DockerHub.txt`.
