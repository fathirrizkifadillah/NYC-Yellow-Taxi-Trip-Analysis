# NYC Yellow Taxi Trip Dataset Explanation

## Dataset Overview

### English

This dataset contains trip-level records from the New York City Taxi & Limousine Commission (NYC TLC) Yellow Taxi service. Each row represents one recorded taxi trip and includes pickup and drop-off timestamps, trip distance, pickup and drop-off location IDs, payment information, fare components, surcharges, and total trip amount.

The dataset is suitable for exploratory data analysis, trip-demand analysis, travel-duration analysis, fare and tipping analysis, route analysis, data-quality assessment, and machine learning workflows such as trip-duration prediction.

### Bahasa Indonesia

Dataset ini berisi data perjalanan taksi kuning dari New York City Taxi & Limousine Commission (NYC TLC). Setiap baris merepresentasikan satu perjalanan taksi yang tercatat, termasuk waktu pickup dan drop-off, jarak perjalanan, ID lokasi pickup dan drop-off, informasi pembayaran, komponen tarif, surcharge, dan total biaya perjalanan.

Dataset ini cocok untuk exploratory data analysis, analisis permintaan perjalanan, analisis durasi perjalanan, analisis tarif dan tip, analisis rute, pemeriksaan kualitas data, serta machine learning seperti prediksi durasi perjalanan.

## Dataset Structure

- **Source:** New York City Taxi & Limousine Commission (NYC TLC)
- **Dataset:** Yellow Taxi Trip Records — January 2026
- **File format:** Parquet
- **Rows:** 3,724,889 raw trip records
- **Original columns:** 20
- **Granularity:** trip-level data
- **Primary time columns:** `tpep_pickup_datetime` and `tpep_dropoff_datetime`
- **Primary location columns:** `PULocationID` and `DOLocationID`
- **Main trip metrics:** `trip_distance`, `fare_amount`, `tip_amount`, and `total_amount`
- **Machine learning target:** `trip_duration_min` *(derived feature)*
- **Important note:** The raw data contains a small number of records outside the January boundary. The analysis should retain trips with pickup timestamps from `2026-01-01` until before `2026-02-01`.

## Column Explanation

| Column | Data Type | English Explanation | Penjelasan Bahasa Indonesia | Analysis Notes |
| --- | --- | --- | --- | --- |
| `VendorID` | Integer | Identifier of the technology vendor that submitted the trip record. | ID vendor teknologi yang mengirimkan record perjalanan. | Useful for comparing trip characteristics and data completeness by vendor. |
| `tpep_pickup_datetime` | Datetime | Date and time when the passenger was picked up. | Tanggal dan waktu saat penumpang dijemput. | Main timestamp for time-based analysis and feature engineering. |
| `tpep_dropoff_datetime` | Datetime | Date and time when the passenger was dropped off. | Tanggal dan waktu saat penumpang diturunkan. | Used with pickup time to calculate trip duration. |
| `passenger_count` | Float | Number of passengers reported for the trip. | Jumlah penumpang yang tercatat dalam perjalanan. | Contains missing values; should be handled carefully. |
| `trip_distance` | Float | Recorded trip distance in miles. | Jarak perjalanan yang tercatat dalam satuan mil. | Useful for distance analysis but contains extreme outliers. |
| `RatecodeID` | Float | Rate code assigned to the trip, such as standard or airport-related rates. | Kode tarif yang diterapkan pada perjalanan, seperti tarif standar atau bandara. | Categorical field with missing values. |
| `store_and_fwd_flag` | String | Indicates whether the trip record was stored before being sent to the system. | Menunjukkan apakah record perjalanan disimpan terlebih dahulu sebelum dikirim ke sistem. | Optional operational field; may be excluded from core analysis. |
| `PULocationID` | Integer | Taxi Zone ID where the passenger was picked up. | ID Taxi Zone tempat penumpang dijemput. | Join with Taxi Zone Lookup to display readable zone names. |
| `DOLocationID` | Integer | Taxi Zone ID where the passenger was dropped off. | ID Taxi Zone tempat penumpang diturunkan. | Join with Taxi Zone Lookup to display readable zone names. |
| `payment_type` | Integer | Payment method category used for the trip. | Kategori metode pembayaran yang digunakan dalam perjalanan. | Useful for analyzing payment behavior and recorded tips. |
| `fare_amount` | Float | Base fare charged for the taxi trip. | Tarif dasar yang dibebankan untuk perjalanan taksi. | Negative values may represent adjustments or refunds. |
| `extra` | Float | Additional charges, such as peak-hour or night-time extras. | Biaya tambahan, misalnya surcharge jam sibuk atau malam hari. | Useful for understanding fare composition. |
| `mta_tax` | Float | Mandatory MTA tax charged for eligible trips. | Pajak MTA yang dikenakan pada perjalanan tertentu. | Part of the total fare composition. |
| `tip_amount` | Float | Tip amount recorded electronically. | Jumlah tip yang tercatat secara elektronik. | Cash tips are not fully captured in the dataset. |
| `tolls_amount` | Float | Total toll charges for the trip. | Total biaya tol selama perjalanan. | Useful for studying long-distance or airport-related trips. |
| `improvement_surcharge` | Float | Surcharge used to support taxi service improvements. | Surcharge untuk mendukung peningkatan layanan taksi. | Part of the total fare composition. |
| `total_amount` | Float | Final amount charged, including fare, tips, tolls, taxes, and surcharges. | Total biaya akhir, termasuk fare, tip, tol, pajak, dan surcharge. | Main transaction-value metric. |
| `congestion_surcharge` | Float | Surcharge related to congestion-pricing rules. | Biaya tambahan terkait aturan congestion pricing. | Contains missing values; inspect before imputing. |
| `Airport_fee` | Float | Additional airport-related charge where applicable. | Biaya tambahan terkait perjalanan bandara jika berlaku. | Useful for airport-trip analysis; contains missing values. |
| `cbd_congestion_fee` | Float | Congestion fee associated with the Central Business District. | Biaya kemacetan yang terkait dengan Central Business District. | Useful for fare-policy and location-related analysis. |

## Data Quality Summary

### English

The raw dataset contains several data-quality considerations:

- Five columns contain missing values: `passenger_count`, `RatecodeID`, `store_and_fwd_flag`, `congestion_surcharge`, and `Airport_fee`.
- Each of these columns has approximately 29.21% missing values.
- Exact duplicate rows were not found in the raw dataset.
- Some records contain negative fare or total amounts. These may represent refunds, corrections, or transaction adjustments.
- Some records have non-positive trip durations and should not be included in normal trip-duration analysis.
- The dataset contains distance and speed outliers that should be inspected before interpretation.

### Bahasa Indonesia

Dataset mentah memiliki beberapa hal yang perlu diperhatikan terkait kualitas data:

- Lima kolom memiliki missing value: `passenger_count`, `RatecodeID`, `store_and_fwd_flag`, `congestion_surcharge`, dan `Airport_fee`.
- Masing-masing kolom tersebut memiliki sekitar 29.21% missing value.
- Tidak ditemukan exact duplicate row pada data mentah.
- Beberapa record memiliki fare atau total amount negatif. Nilai ini dapat merepresentasikan refund, correction, atau adjustment transaksi.
- Beberapa record memiliki durasi perjalanan nol atau negatif dan tidak boleh digunakan untuk analisis durasi perjalanan normal.
- Dataset memiliki outlier pada jarak dan kecepatan perjalanan yang perlu diperiksa sebelum menarik kesimpulan.

## Data Cleaning Rules

### English

The following cleaning rules are applied to prepare the data for EDA and machine learning:

1. Create `trip_duration_min` from pickup and drop-off timestamps.
2. Remove records with non-positive trip durations.
3. Remove records with negative `fare_amount` or negative `total_amount`.
4. Retain records with missing values unless the affected variable is required for a specific analysis or model.
5. Handle missing categorical variables selectively using an `Unknown` category where appropriate.
6. Inspect extreme values in trip distance, speed, fare, and tips before deciding whether additional outlier filtering is needed.
7. Restrict the analytical period to January 2026 using the pickup timestamp.

### Bahasa Indonesia

Aturan cleaning berikut diterapkan untuk menyiapkan data bagi EDA dan machine learning:

1. Membuat `trip_duration_min` dari timestamp pickup dan drop-off.
2. Menghapus record dengan durasi perjalanan nol atau negatif.
3. Menghapus record dengan `fare_amount` atau `total_amount` negatif.
4. Mempertahankan record yang memiliki missing value kecuali kolom tersebut diperlukan untuk analisis atau model tertentu.
5. Menangani missing value kategorikal secara selektif dengan kategori `Unknown` jika sesuai.
6. Memeriksa nilai ekstrem pada jarak perjalanan, kecepatan, fare, dan tip sebelum memutuskan apakah filtering outlier tambahan diperlukan.
7. Membatasi periode analisis pada Januari 2026 berdasarkan waktu pickup.

## Useful Derived Features

### English

You can create the following derived features after cleaning the dataset:

| Derived Feature | Formula / Idea | Purpose |
| --- | --- | --- |
| `trip_duration_min` | `tpep_dropoff_datetime - tpep_pickup_datetime` in minutes | Main trip-duration metric and target variable for machine learning. |
| `pickup_date` | Extract date from `tpep_pickup_datetime` | Useful for daily trip-volume analysis. |
| `pickup_hour` | Extract hour from `tpep_pickup_datetime` | Useful for hourly demand and travel-duration analysis. |
| `pickup_dayofweek` | Extract weekday number, Monday = 0 to Sunday = 6 | Useful for comparing weekday patterns. |
| `pickup_day_name` | Extract weekday name from pickup timestamp | Makes charts and reports easier to read. |
| `is_weekend` | `1` for Saturday/Sunday, otherwise `0` | Useful for comparing weekday and weekend trips. |
| `is_rush_hour` | Flag trips during selected morning and evening peak hours | Useful for congestion and rush-hour analysis. |
| `is_night` | Flag trips between 00:00 and 05:59 | Useful for late-night demand analysis. |
| `pickup_period` | Group pickup hours into Late Night, Morning, Afternoon, Evening, and Night | Creates cleaner time-of-day categories for visualization. |
| `route` | Combine `PULocationID` and `DOLocationID` | Useful for identifying popular origin-destination routes. |
| `distance_category` | Group `trip_distance` into distance bands | Useful for comparing short, medium, and long trips. |
| `speed_mph` | `trip_distance / trip_duration_hours` | Useful for traffic analysis and detecting extreme speed outliers. |
| `tip_percentage` | `(tip_amount / fare_amount) * 100` | Useful for analyzing recorded electronic tipping behavior. |
| `fare_per_mile` | `fare_amount / trip_distance` | Useful for comparing fare efficiency across trip distances. |

### Bahasa Indonesia

Kamu dapat membuat fitur turunan berikut setelah proses data cleaning:

| Fitur Turunan | Formula / Ide | Tujuan |
| --- | --- | --- |
| `trip_duration_min` | `tpep_dropoff_datetime - tpep_pickup_datetime` dalam menit | Metrik utama durasi perjalanan dan target untuk machine learning. |
| `pickup_date` | Mengambil tanggal dari `tpep_pickup_datetime` | Berguna untuk analisis jumlah trip harian. |
| `pickup_hour` | Mengambil jam dari `tpep_pickup_datetime` | Berguna untuk analisis permintaan dan durasi perjalanan per jam. |
| `pickup_dayofweek` | Mengambil angka hari, Senin = 0 sampai Minggu = 6 | Berguna untuk membandingkan pola perjalanan berdasarkan hari. |
| `pickup_day_name` | Mengambil nama hari dari waktu pickup | Membuat chart dan laporan lebih mudah dibaca. |
| `is_weekend` | `1` untuk Sabtu/Minggu, selain itu `0` | Berguna untuk membandingkan trip weekday dan weekend. |
| `is_rush_hour` | Penanda trip pada jam sibuk pagi dan sore | Berguna untuk analisis kemacetan dan pola rush hour. |
| `is_night` | Penanda trip antara pukul 00:00–05:59 | Berguna untuk analisis permintaan perjalanan malam hari. |
| `pickup_period` | Mengelompokkan jam pickup menjadi Late Night, Morning, Afternoon, Evening, dan Night | Membuat kategori waktu yang lebih rapi untuk visualisasi. |
| `route` | Menggabungkan `PULocationID` dan `DOLocationID` | Berguna untuk mengidentifikasi rute asal-tujuan yang populer. |
| `distance_category` | Mengelompokkan `trip_distance` ke dalam kategori jarak | Berguna untuk membandingkan trip pendek, sedang, dan jauh. |
| `speed_mph` | `trip_distance / trip_duration_hours` | Berguna untuk analisis lalu lintas dan mendeteksi outlier kecepatan. |
| `tip_percentage` | `(tip_amount / fare_amount) * 100` | Berguna untuk analisis perilaku tip elektronik yang tercatat. |
| `fare_per_mile` | `fare_amount / trip_distance` | Berguna untuk membandingkan efisiensi tarif berdasarkan jarak perjalanan. |

## Suggested Analysis Questions

### English

1. At what hours does Yellow Taxi demand reach its peak?
2. How do trip volume and average trip duration differ between weekdays and weekends?
3. Which pickup and drop-off zones generate the most trips?
4. Which routes are most frequently used by passengers?
5. What is the distribution of trip duration and trip distance?
6. Are rush-hour trips slower than non-rush-hour trips?
7. How do average fare, total amount, and tips vary by hour of day?
8. Which payment method is used most frequently?
9. Are trips with longer distances associated with higher tips or higher fare per mile?
10. What percentage of records contains missing values, invalid durations, or negative monetary values?
11. Can trip duration be predicted from pickup time, pickup location, drop-off location, vendor, and passenger-related information?

### Bahasa Indonesia

1. Pada jam berapa permintaan Yellow Taxi paling tinggi?
2. Bagaimana perbedaan jumlah trip dan rata-rata durasi perjalanan antara weekday dan weekend?
3. Zona pickup dan drop-off mana yang menghasilkan jumlah trip tertinggi?
4. Rute mana yang paling sering digunakan penumpang?
5. Bagaimana distribusi durasi perjalanan dan jarak perjalanan?
6. Apakah perjalanan saat rush hour lebih lambat daripada jam biasa?
7. Bagaimana rata-rata fare, total amount, dan tip berubah berdasarkan jam?
8. Metode pembayaran apa yang paling sering digunakan?
9. Apakah trip dengan jarak lebih jauh berkaitan dengan tip lebih tinggi atau fare per mile lebih tinggi?
10. Berapa persentase record yang memiliki missing value, durasi tidak valid, atau nominal negatif?
11. Apakah durasi perjalanan dapat diprediksi menggunakan waktu pickup, lokasi pickup, lokasi drop-off, vendor, dan informasi penumpang?

## Machine Learning Scope

### English

The machine learning objective is to predict `trip_duration_min`.

Recommended input features:

- `pickup_hour`
- `pickup_dayofweek`
- `is_weekend`
- `is_rush_hour`
- `VendorID`
- `PULocationID`
- `DOLocationID`
- `passenger_count`
- `RatecodeID`

The model should use a time-based train/test split. Post-trip fields such as `fare_amount`, `total_amount`, `tip_amount`, `payment_type`, `speed_mph`, and actual `trip_distance` should not be used as model inputs because they can cause data leakage.

### Bahasa Indonesia

Tujuan machine learning dalam proyek ini adalah memprediksi `trip_duration_min`.

Fitur input yang direkomendasikan:

- `pickup_hour`
- `pickup_dayofweek`
- `is_weekend`
- `is_rush_hour`
- `VendorID`
- `PULocationID`
- `DOLocationID`
- `passenger_count`
- `RatecodeID`

Model sebaiknya menggunakan time-based train/test split. Kolom yang baru diketahui setelah perjalanan selesai seperti `fare_amount`, `total_amount`, `tip_amount`, `payment_type`, `speed_mph`, dan `trip_distance` aktual tidak boleh digunakan sebagai input model karena dapat menyebabkan data leakage.

## Important Notes

### English

- Each row represents a taxi trip, not a unique passenger or a unique driver.
- The dataset does not include a dedicated trip ID, so exact duplicate checks are performed at the row level.
- `tip_amount` mainly reflects electronically recorded tips. Cash tips may not appear in the dataset.
- `PULocationID` and `DOLocationID` should ideally be joined with the NYC TLC Taxi Zone Lookup table to display actual zone names.
- This dataset covers one primary month, so it is suitable for within-month demand analysis but limited for long-term seasonal forecasting.
- Insights should be described as observational findings and should not automatically be treated as causal conclusions.

### Bahasa Indonesia

- Setiap baris merepresentasikan satu perjalanan taksi, bukan satu penumpang unik atau satu pengemudi unik.
- Dataset tidak memiliki ID trip khusus, sehingga pengecekan duplicate dilakukan pada level seluruh baris.
- `tip_amount` terutama mencerminkan tip elektronik. Tip tunai kemungkinan tidak terekam dalam dataset.
- `PULocationID` dan `DOLocationID` idealnya di-join dengan NYC TLC Taxi Zone Lookup agar nama zonanya lebih mudah dibaca.
- Dataset ini mencakup satu bulan utama, sehingga cocok untuk analisis permintaan dalam satu bulan tetapi terbatas untuk forecasting musiman jangka panjang.
- Insight yang diperoleh bersifat observasional dan tidak boleh langsung dianggap sebagai hubungan sebab-akibat.