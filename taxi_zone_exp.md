# NYC Taxi Zone Lookup Dataset Explanation

## Dataset Overview

### English

This dataset is a reference table provided by the New York City Taxi & Limousine Commission (NYC TLC). It maps Taxi Zone IDs to readable geographic information, including borough, zone name, and service zone.

Unlike the Yellow Taxi Trip dataset, each row in this file does not represent a taxi trip. Instead, each row represents one Taxi Zone that can be used to translate `PULocationID` and `DOLocationID` from the trip dataset into meaningful location names.

### Bahasa Indonesia

Dataset ini adalah tabel referensi dari New York City Taxi & Limousine Commission (NYC TLC). Dataset ini memetakan Taxi Zone ID ke informasi geografis yang mudah dibaca, seperti borough, nama zona, dan service zone.

Berbeda dengan dataset Yellow Taxi Trip, setiap baris pada file ini bukan merepresentasikan satu perjalanan taksi. Setiap baris merepresentasikan satu Taxi Zone yang digunakan untuk menerjemahkan `PULocationID` dan `DOLocationID` dari dataset trip menjadi nama lokasi yang lebih mudah dipahami.

## Dataset Structure

- **Source:** New York City Taxi & Limousine Commission (NYC TLC)
- **File name:** `taxi_zone_lookup.csv`
- **File format:** CSV
- **Rows:** 265 Taxi Zone records
- **Columns:** 4
- **Granularity:** taxi-zone-level reference data
- **Primary key:** `LocationID`
- **Main purpose:** Enrich Yellow Taxi Trip data with borough and zone names
- **Duplicate rows:** 0
- **Unique `LocationID`:** 265

## Column Explanation

| Column | Data Type | English Explanation | Penjelasan Bahasa Indonesia | Analysis Notes |
| --- | --- | --- | --- | --- |
| `LocationID` | Integer | Unique identifier for each NYC Taxi Zone. | ID unik untuk setiap Taxi Zone di NYC. | Used to join with `PULocationID` and `DOLocationID` in the trip dataset. |
| `Borough` | String | Borough where the Taxi Zone is located, such as Manhattan, Queens, Brooklyn, Bronx, or Staten Island. | Wilayah borough tempat Taxi Zone berada, seperti Manhattan, Queens, Brooklyn, Bronx, atau Staten Island. | Useful for borough-level trip demand and destination analysis. |
| `Zone` | String | Readable name of the Taxi Zone. | Nama Taxi Zone yang mudah dibaca manusia. | Main field for readable charts, route analysis, and reporting. |
| `service_zone` | String | Taxi service classification for the zone. | Klasifikasi layanan taksi untuk zona tersebut. | Useful for distinguishing Yellow Taxi zones, Boro zones, EWR, and other service areas. |

## Data Quality Summary

### English

- The lookup table contains 265 unique Taxi Zone IDs.
- No duplicate rows were found.
- `LocationID` is complete and can be used as a reliable join key.
- A very small number of records contain missing geographic labels, so these should be preserved and displayed as `Unknown` if needed.

### Bahasa Indonesia

- Tabel lookup memiliki 265 Taxi Zone ID yang unik.
- Tidak ditemukan duplicate row.
- `LocationID` lengkap dan dapat digunakan sebagai join key yang andal.
- Sejumlah sangat kecil record memiliki label geografis yang kosong, sehingga dapat dipertahankan dan ditampilkan sebagai `Unknown` jika diperlukan.

## How This Dataset Is Used

### English

The lookup table should be joined twice with the Yellow Taxi Trip dataset:

1. Join `PULocationID` with `LocationID` to obtain pickup borough and pickup zone.
2. Join `DOLocationID` with `LocationID` to obtain drop-off borough and drop-off zone.

This enrichment makes the analysis much more understandable. For example, instead of reporting that location ID `237` has the highest trip volume, the analysis can report the actual zone name.

### Bahasa Indonesia

Tabel lookup perlu di-join dua kali dengan dataset Yellow Taxi Trip:

1. Join `PULocationID` dengan `LocationID` untuk mendapatkan borough dan zona pickup.
2. Join `DOLocationID` dengan `LocationID` untuk mendapatkan borough dan zona drop-off.

Proses enrichment ini membuat analisis jauh lebih mudah dipahami. Contohnya, daripada hanya menyebut location ID `237` memiliki jumlah trip tertinggi, analisis dapat menyebut nama zona sebenarnya.

## Suggested Join Logic

```python
zone_lookup = pd.read_csv("data/taxi_zone_lookup.csv")
