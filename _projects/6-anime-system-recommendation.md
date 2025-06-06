---
title: "Anime Recommender"
excerpt: "Proyek ini mengembangkan dua sistem rekomendasi tontonan Anime dengan pendekatan yang berbeda, yakni Content-Based Filtering dan Collaborative Filtering. Sistem Content-Based menyajikan rekomendasi berdasarkan kesamaan fitur seperti genre, sinopsis, dan skor, sementara sistem Collaborative Filtering menganalisis pola rating pengguna lain dengan preferensi serupa. Proyek ini bertujuan untuk membantu pengguna menemukan anime yang sesuai dengan minat mereka secara lebih akurat dan efisien."
date: 2025-05-18
author_profile: false
---

# Tools

- **Jupyter Notebook**: [Colab](https://github.com/camelliatea/anime-recommendation-system/blob/main/System_Recommendation_Notebook.ipynb)
- **Bahasa Pemrograman:** Python
- **Pustaka:** 
    - Pandas dan Numpy, untuk manipulasi dan analisis data.
    - Matplotlib, untuk visualisasi data.
    - scikit-learn, untuk ekstraksi fitur teks (TF-IDF) dan perhitungan cosine similarity dalam pendekatan Content-Based Filtering.
    - TensorFlow dan Keras, untuk pembangunan dan pelatihan model machine learning dalam pendekatan Collaborative Filtering.
- **Sumber Dataset:** [Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)

# Project Domain

Seiring meningkatkan jumlah konten hiburan yang tersedia secara online, termasuk anime (animasi dari Jepang), pengguna menghadapi tantangan dalam menemukan konten yang relevan dan sesuai dengan preferensi mereka. Situasi ini merupakan salah satu cerminan dari fenomena information overload, di mana pengguna mengalami kesulitan dalam memilah informasi yang sesuai dengannya dari sekian banyak pilihan yang tersedia. Adapun, tujuan dari proyek ini adalah membangun sistem rekomendasi anime ayang efektif dengan memanfaatkan dataset hasil web scraping dari situs komunitas anime terbesar, yakni [MyAnimeList](https://myanimelist.net/anime.php), yang mencakup berbagai informasi penting sebagai dasar dalam proses rekomendasi, termasuk judul, genre, jumlah episode, serta rating dari pengguna. Harapannya, pengguna dapat menemukan anime yang relevan melalui pendekatan personalisasi berbasis data. 

Sistem rekomendasi yang dikembangkan menggabungkan dua pendekatan utama. Pertama, content-based filtering yang menggunakan genre sebagai fitur utama untuk merekomendasikan anime dengan karakteristik serupa dengan yang disukai pengguna sebelumnya. Kedua, collaborative filtering yang menganalisis pola rating antarpengguna untuk mengidentifikasi kesamaan preferensi sehingga dapat merekomendasikan anime yang belum ditonton oleh pengguna tetapi kemungkinan besar akan disukai. Dengan pendekatan ini, sistem dapat memberikan rekomendasi yang lebih personal dan sesuai dengan kebiasaan konsumsi pengguna.

Selaras dengan studi oleh Isinkaye, Folajimi, dan Ojokoh (2015), sistem rekomendasi membuka peluang baru dalam menyajikan informasi yang personal dan efektif kepada pengguna utnuk mengatasi *information overload*. Studi tersebut juga menyoroti kekuatan dan tantangan dari teknik rekomendasi tradisional, serta pentingnya strategi hybridisasi dan pemilihan algoritma pembelajaran yang tepat untuk meningkatkan kualitas rekomendasi. Dengan landasan ini, proyek bertujuan menciptakan sistem yang tidak hanya relevan secara personal, tetapi juga mampu belajar dan beradaptasi dari data pengguna secara berkelanjutan.

Referensi:
- Isinkaye, F. O., Folajimi, Y. O., & Ojokoh, B. A. (2015). Recommendation systems: Principles, methods and evaluation. Egyptian informatics journal, 16(3), 261-273.

# Business Understanding

Dengan mempertimbangkan kebutuhan pengguna dalam menghadapi banyaknya pilihan tontonan anime, serta potensi yang dimiliki oleh data dari situs komunitas MyAnimeList, proyek ini diarahkan untuk memberikan solusi berbasis personalisasi guna mempermudah proses pencarian konten yang relevan.


## Problem Statements

- Bagaimana cara membantu pengguna menemukan tontonan anime yang relevan dan sesuai preferensi pribadi mereka di tengah jumlah konten yang sangat banyak (information overload)?
- Bagaimana mengatasi keterbatasan sistem rekomendasi yang hanya menggunakan satu pendekatan agar hasil rekomendasinya lebih optimal dan adaptif?

## Goals

- Membangun sistem rekomendasi yang mampu mengurangi beban information overload dengan memberikan saran anime yang telah dipersonalisasi berdasarkan preferensi pengguna.
- Menerapkan kombinasi dua pendekatan utama, yaitu content-based filtering dan collaborative filtering, untuk mengatasi keterbatasan sistem rekomendasi tunggal dan meningkatkan kualitas hasil rekomendasi.

## Solution statements

- **Solution Approach 1: Content-Based Filtering**

    Menggunakan informasi genre dari setiap anime untuk membangun model rekomendasi yang menyarankan anime dengan kesamaan karakteristik konten terhadap anime yang sebelumnya disukai pengguna. Pendekatan ini mempersonalisasi hasil berdasarkan item.

- **Solution Approach 2: Collaborative Filtering**

    Menerapkan algoritma berbasis rating pengguna untuk mengidentifikasi pola preferensi yang mirip antara pengguna satu dengan lainnya sehingga dapat merekomendasikan anime yang belum ditonton namun disukai oleh pengguna dengan preferensi serupa.


# Data Understanding

Dataset yang digunakan dalam proyek ini adalah [Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database/data), hasil kompilasi data dari situs komunitas anime terbesar, yaitu *MyAnimeList*, yang dikumpulkan oleh pengguna kaggle *CooperUnion* melalui pengolahan dari API publik. Dataset ini menyajikan informasi mengenai preferensi pengguna terhadap anime yang mencakup data dari 73.516 pengguna dan 12.294 judul anime dalam dua berkas utama, yaitu `anime.csv` dan `rating.csv`. Berikut ini adalah variabel yang terkandung di dalam kedua berkas tersebut.

## Variabel Description

### Anime.csv

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 12294 entries, 0 to 12293
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   anime_id  12294 non-null  int64  
 1   name      12294 non-null  object 
 2   genre     12232 non-null  object 
 3   type      12269 non-null  object 
 4   episodes  12294 non-null  object 
 5   rating    12064 non-null  float64
 6   members   12294 non-null  int64  
dtypes: float64(1), int64(2), object(4)
memory usage: 672.5+ KB
```

- Dataset ini terdiri dari 7 kolom dan 12.294 baris, dengan 1 kolom bertipe float64, 2 kolom bertipe int64, dan 4 kolom bertipe object.
- Kolom `anime_id` memiliki tipe data int64 dengan total 12.294 sampel data, yang merepresentasikan ID unik untuk mengindentifikasi setiap anime.
- Kolom `name` memiliki tipe data object dengan total 12.294 sampel data, yang  judul dari setiap anime.
- Kolom `genre` memiliki tipe data object dengan total 12.232 sampel data, yang merepresetasikan daftar genre dari setiap anime.
- Kolom `type` memiliki tipe data object dengan total 12.269 sampel data, yang merepresentasikan jenis penyajian setiap anime.
- Kolom `episodes` memiliki tipe data object dengan total 12.294 sampe data, yang merepresentasikan jumlah episode dari setiap anime.
- Kolom `rating` memiliki tipe data float64 dengan total 12.064 sampel data, yang merepresentasikan rata-rata rating dari komunitas dalam skala 1 sampai 10.
- Kolom `members` memiliki tipe data int64 dengan total 12.294 sampel data, yang merepresentasikan jumlah anggota komunitas MyAnimeList yang memasukkan anime tertentu ke dalam daftar mereka.

### Rating.csv

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7813737 entries, 0 to 7813736
Data columns (total 3 columns):
 #   Column    Dtype
---  ------    -----
 0   user_id   int64
 1   anime_id  int64
 2   rating    int64
dtypes: int64(3)
memory usage: 178.8 MB
```

- Dataset ini terdiri dari 3 kolom dan 7..813.737 baris data, yang mana seluruh data bertipe int64.
- Kolom `user_id` merepresentasikan ID unik untuk pengguna yang bersifat anonim dan dihasilkan secara acak.
- Kolom `anime_id` merepresentasikan ID unik untuk mengidentifikasi setiap anime, berperan sebagai foreign key untuk menghubungkan file `rating.csv` dengan `anime.csv`.
- Kolom `rating` merepresentasikan ilai rating yang diberikan setiap pengguna kepada suatu anime. Nilainya berkisar antara 1 sampai 10, dengan -1 jika pengguna menonton anime tersebut tetapi tidak memberikan rating.

## Univariate Exploratory Analysis

```
Jumlah Anime: 12294
Jumlah genre:  43
Daftar genre:  {'Music', 'Cars', 'Samurai', 'Adventure', 'Dementia', 'Thriller', 'Shounen Ai', 'Slice of Life', 'Harem', 'Action', 'Sci-Fi', 'Shounen', 'Shoujo', 'Military', 'Kids', 'Demons', 'Game', 'Comedy', 'Sports', 'Historical', 'Police', 'Parody', 'Mystery', 'Ecchi', 'Martial Arts', 'Yaoi', 'School', 'Shoujo Ai', 'Super Power', 'Space', 'Mecha', 'Vampire', 'Seinen', 'Josei', 'Yuri', 'Drama', 'Magic', 'Hentai', 'Psychological', 'Supernatural', 'Horror', 'Fantasy', 'Romance'}
Jumlah Tipe: 7
Daftar Tipe: ['Movie' 'TV' 'OVA' 'Special' 'Music' 'ONA' nan]

|           | Episodes | Rating | Members       |
|-----------|----------|--------|---------------|
| Count     | 11,954   | 12,064 | 12,294        |
| Mean      | 12.38    | 6.47   | 18,071.34     |
| Std       | 46.87    | 1.03   | 54,820.68     |
| Min       | 1        | 1.67   | 5             |
| 25%       | 1        | 5.88   | 225           |
| 50%       | 2        | 6.57   | 1,550         |
| 75%       | 12       | 7.18   | 9,437         |
| Max       | 1,818    | 10.00  | 1,013,917     |

|           | Rating     |
|-----------|------------|
| Count     | 7,813,737  |
| Mean      | 6.14       |
| Std       | 3.73       |
| Min       | -1.00      |
| 25%       | 6.00       |
| 50%       | 7.00       |
| 75%       | 9.00       |
| Max       | 10.00      |

```

Dari eksplorasi di atas, diketahui beberapa informasi berikut:

- Dataset mengandung 12.294 judul anime unik.
- Anime dalam dataset dikategorikan ke dalam 7 tipe unik, yaitu Movie, TV, OVA, Special, Music, ONA, dan nan. 'nan' di sini mengacu pada nilai kosong sehingga akan dibersihkan pada tahapan selanjutnya.
- Anime dalam dataset tersebar ke dalam 43 tipe genre, seperti Romance, Sci-Fi, Historical, dan banyak lagi.
- Jumlah episode pada setiap anime bervariasi dari 1 hingga 1.818 (anime Oyako Club) dengan rata-rata 12 episode dan standar deviasi sebesar 46.87, yang mana cukup besar dan menunjukkan adanya banyak anime berdurasi pendek serta beberapa anime dengan episode yang sangat panjang (outlier).
- Rating komunitas terhadap anime berkisar antara 1,67 hingga 10, dengan rata-rata 6,47 dan standar deviasi sebesar 1,03 yang mana cukup kecil dan mengindikasikan bahwa sebagain besar anime memiliki penilaian yan cukup baik dan relatif homogen.
- Jumlah anggota komunitas untuk setiap judul anime berkisar antara 5 hingga lebih dari 1 juta orang, dengan rata-rata 18.071 anggota dan standar deviasi sebesar 54,820, yang mana cukup besar dan menunjukkan adanya ketimpangan popularitas yang besar antarjudul.
- User rating berada dalam kisaran -1 hingga 10, di mana nilai -1 menandakan bahwa pengguna telah menonton anime namun tidak memberikan rating, sehingga nilai ini perlu diperlakukan sebagai data tidak valid atau missing value dalam proses analisis lebih lanjut. Rata-rata rating yang diberikan pengguna adalah 6,14 dengan standar deviasi sebesar 3,73, yang menunjukkan adanya variasi penilaian yang cukup besar di antara pengguna terhadap anime yang mereka tonton.

# Data Preparation

Tahapan ini bertujuan untuk membersihkan dan menyiapkan data dari dua dataset utama, yaitu anime.csv dan rating.csv, agar dapat digunakan secara optimal dalam model rekomendasi. Proses ini dibagi ke dalam tiga tahapan penting sebagai berikut:

## Data Preprocessing

Tahapan ini mencakup tiga proses utama, yaitu penggabungan dua dataset (anime dan rating) disertai dengan penyesuaian nama kolom yang serupa untuk menghindari kebingungan, pemeriksaan konsistensi skala rating agar seluruh nilai berada dalam rentang valid 1 hingga 10, serta penanganan nilai yang hilang (missing values) untuk memastikan integritas serta kualitas data yang akan digunakan dalam proses pembangunan model.

1. **Merge and Transform Data**

    Penggabungan data dilakukan menggunakan `pd.merge` dengan metode `left` join berdasarkan kolom `anime_id`. Metode `left` join dipilih untuk mempertahankan seluruh data dari dataset ratings yang berisi aktivitas pengguna, meskipun beberapa `anime_id` mungkin tidak memiliki pasangan yang cocok di dataset anime. Setelah penggabungan, karena kedua dataset memiliki kolom rating, pandas secara otomatis menamainya sebagai `rating_x` (dari `ratings`) dan `rating_y` (dari `anime`). Oleh karena itu, kolom `rating_x` diubah menjadi `user_rating` dan `rating_y` menjadi `community_rating` dengan fungsi rename untuk memperjelas perbedaan. Langkah ini penting untuk memastikan integritas dan keterbacaan data yang akan digunakan dalam pemodelan sistem rekomendasi. Selain itu, dilakukan juga pengubahan tipe data untuk fitur `user_rating` yang mulanya int64 menjadi float64.

2. **User Rating Scaling**

    Untuk memastikan bahwa skala rating berada pada rentang valid antara 1 hingga 10, dilakukan proses filtering pada DataFrame `anime_ratings` menggunakan kondisi boolean (`(anime_ratings['user_rating'] >= 1) & (anime_ratings['user_rating'] <= 10)`) sehingga hanya baris-baris dengan nilai `user_rating` di antara 1 dan 10 (True) yang dipertahankan, sementara nilai di luar rentang tersebut (False) dikeluarkan dari data untuk menjaga konsistensi dan kualitas input bagi model.
    Hasilnya:

        |           | user_rating |
        |-----------|-------------|
        | count     | 6,337,241   |
        | mean      | 7.808497    |
        | std       | 1.572496    |
        | min       | 1.000000    |
        | 25%       | 7.000000    |
        | 50%       | 8.000000    |
        | 75%       | 9.000000    |
        | max       | 10.000000   |

    - Rating pengguna sudah berada dalam rentang yang valid, yaitu 1 hingga 10, dengan rata-rata rating sebesar 7,8 dan standar deviasi sebesar 1,5, menunjukkan bahwa mayoritas rating cenderung positif dan konsisten

3. **Missing Values Handling**

    Untuk menangani nilai kosong pada dataset, pertama-tama dilakukan pengecekan dengan menggunakan `isnull().sum()` Untuk menangani nilai kosong pada dataset, pertama-tama dilakukan pengecekan dengan menggunakan `dropna()`, dan hasil pembersihan disimpan ke dalam variabel `anime_cleaned`. Untuk memastikan bahwa proses pembersihan berhasil dan tidak ada nilai kosong yang tersisa, dilakukan pengecekan ulang menggunakan `isnull().sum()`.

Adapun, struktur dataset setelah seluruh proses dilakukan sebagai berikut:

```
<class 'pandas.core.frame.DataFrame'>
Index: 6337241 entries, 47 to 7813736
Data columns (total 9 columns):
 #   Column            Dtype  
---  ------            -----  
 0   user_id           int64  
 1   anime_id          int64  
 2   user_rating       float64  
 3   name              object 
 4   genre             object 
 5   type              object 
 6   episodes          float64
 7   community_rating  float64
 8   members           float64
dtypes: float64(3), int64(3), object(3)
memory usage: 483.5+ MB

Jumlah Anime dalam dataset:  9890
Jumlah Tipe Anime dalam dataset:  6
Jumlah genre:  43
Daftar genre:  {'Samurai', 'Shoujo Ai', 'Vampire', 'Music', 'Space', 'Yaoi', 'Dementia', 'Police', 'Shounen', 'Military', 'Romance', 'Comedy', 'Seinen', 'Harem', 'Magic', 'Supernatural', 'Mecha', 'Drama', 'Action', 'Sports', 'Super Power', 'Slice of Life', 'Shoujo', 'Game', 'Psychological', 'Cars', 'Josei', 'Fantasy', 'Thriller', 'Mystery', 'Martial Arts', 'Adventure', 'Historical', 'School', 'Horror', 'Ecchi', 'Kids', 'Yuri', 'Parody', 'Sci-Fi', 'Hentai', 'Shounen Ai', 'Demons'}
```

- Total sampel data yang tersedia saat ini berjumlah 6.337.241 baris dengan 9 kolom, terdiri dari 4 kolom bertipe float64 (`episodes`, `community_rating`, `members`, `user_rating`), 2 kolom bertipe int64 (`user_id`, `anime_id`), dan 3 kolom bertipe object (`name`, `genre`, `type`).
- Jumlah anime unik berkurang dari 12.294 menjadi 9.890 setelah proses pembersihan data, kategori tipe anime menyusut dari 7 menjadi 6, sedangkan jumlah daftar genre tetap konsisten yaitu sebanyak 43 genre.


## Data Preparation for Content-based Filtering

Pada tahapan ini, data anime yang sudah dibersihkan (`anime_cleaned`) disalin ke dalam variabel `preparation_content` untuk diproses lebih lanjut sebagai dasar pembuatan model content-based filtering. Data tersebut kemudian diurutkan berdasarkan `anime_id` menggunakan `sort_values()` dan dilakukan penghapusan duplikasi berdasarkan kolom `anime_id` menggunakan `drop_duplicates()` untuk memastikan setiap anime hanya muncul sekali.

Selanjutnya, beberapa kolom penting seperti `anime_id`, `name`, `genre`, dan `type` dikonversi dari format series menjadi list menggunakan `tolist()` agar mudah digunakan dalam proses pembentukan fitur. Panjang dari masing-masing list tersebut juga diperiksa untuk memastikan konsistensi data.

Terakhir, data dari list-list tersebut digabungkan kembali ke dalam sebuah DataFrame baru bernama `anime_new` yang hanya berisi kolom `id` (anime_id), `anime_name`, `genre`, dan `type`. DataFrame ini akan menjadi dasar dalam pembuatan sistem rekomendasi content-based filtering yang memanfaatkan atribut genre anime sebagai fitur utama. Contoh hasilnya sebagai berikut:

| id | anime_name                  | genre                                                 | type  |
|----|-----------------------------|--------------------------------------------------------|-------|
| 1  | Cowboy Bebop                | Action, Adventure, Comedy, Drama, Sci-Fi, Space       | TV    |
| 5  | Cowboy Bebop: Tengoku no Tobira | Action, Drama, Mystery, Sci-Fi, Space             | Movie |
| 6  | Trigun                      | Action, Comedy, Sci-Fi                                | TV    |
| 7  | Witch Hunter Robin          | Action, Drama, Magic, Mystery, Police, Supernatural   | TV    |
| 8  | Beet the Vandel Buster      | Adventure, Fantasy, Shounen, Supernatural             | TV    |

## Data Preparation for Collaborative Filtering

Pada tahapan ini, dilakukan dua proses utama, yaitu persiapan data rating khusus untuk model Collaborative Filtering dan pembagian data menjadi data pelatihan (training) dan data validasi (validation). Seluruh proses ini dilakukan menggunakan variabel `preparation_collab` yang merupakan salinan dari dataset `anime_cleaned` agar proses manipulasi tidak mengubah data asli. Langkah-langkah ini memastikan bahwa data sudah dalam format numerik, terstruktur, dan teracak dengan baik sehingga siap digunakan sebagai input untuk model rekomendasi berbasis rating

1. **Rating Data Preparation**

    Persiapan data rating dilakukan dalam beberapa langkah berikut:

    - Encoding Fitur

        Dalam proses ini, Fitur `user_id` dan `anime_id` dikodekan (encoded) menjadi indeks integer berurutan. Encoding ini penting karena algoritma rekomendasi umumnya lebih efisien dan kompatibel dengan indeks numerik daripada menggunakan ID asli yang bersifat acak atau berupa string.

        Proses dimulai dengan mengambil daftar unik `user_id` dan `anime_id` dari dataset, kemudian dibuatlah dictionary pemetaan dari `user_id` asli ke indeks integer (`user_to_user_encoded`) serta pemetaan sebaliknya (`user_encoded_to_user`). Pemetaan balik ini berguna untuk mengonversi kembali hasil prediksi model ke ID asli saat interpretasi output. Langkah yang sama dilakukan pada `anime_id` dengan membuat dictionary `anime_to_anime_encoded` dan `anime_encoded_to_anime`.

    - Pemetaan ke Dataframe

        Setelah dictionary dibuat, kolom `user` dan `anime` ditambahkan ke dalam Dataframe dengan memanfaatkan fungsi `.map()` dari pandas untuk mengganti nilai `user_id` dan `anime_id` dengan indeks hasil encoding.

    - Pengecekan Skala dan Validasi
        
        Selanjutnya, dilakukan pengecekan jumlah user dan anime unik dengan fungsi  `len() ` pada dictionary encoding untuk mengetahui skala dataset yang digunakan, serta fungsi `min()` dan `max()` digunakan untuk memastikan skala rating mulai dari 1.0 hingga 10.0.

    Dari proses tersebut, diketahui bahwa  `preparation_collab` mencakup 69.600 pengguna unik dan 9.890 anime unik, dengan skala rating 1.0 – 10.0, dan sudah siap untuk dimasukkan ke dalam model Collaborative Filtering.

2. **Train-Val Split**

    Setelah data dipersiapkan, dilakukan pembagian dataset menjadi data pelatihan dan validasi untuk melatih serta menguji performa model:

    - Mengacak Data

        Dataset diacak menggunakan` .sample(frac=1, random_state=42)` untuk memastikan distribusi data acak namun tetap konsisten saat dijalankan kembali karena `random_state`.

    - Membentuk Data Fitur dan Target

        `x` = pasangan fitur yang terdiri dari user dan anime, diperoleh dengan `[['user', 'anime']].values`.
        `y` = target berupa rating yang telah dinormalisasi ke rentang 0–1 menggunakan formula:
        $$ y = \frac{\text{rating} - \text{min\_rating}}{\text{max\_rating} - \text{min\_rating}} $$
        Normalisasi ini penting agar model bekerja lebih stabil dan cepat konvergen saat proses pelatihan.

    - Membagi Dataa 80:20

        Dataset dibagi menjadi (1) `x_train` dan `y_train` sebanyak 80%, dan (2) `x_val` dan `y_val` sebanyak 20%

# Modeling

Pada tahapan ini, digunakan dua pendekatan utama untuk mengembangkan sistem rekomendasi, yaitu *Conntent-Based Filtering* dan *Collaborative Filtering.* Kedua pendekatan ini diterapkan untuk menghasilkan rekomendasi yang revelan dan sesuai dengan preferensi pengguna sesuai dengan data yang tersedia.

## Model Development - Content-Based Filtering

Genre digunakan sebagai fitur utama untuk mengembangkan model Content-Based Filtering. Ini dipilih karena genre secara langsung merepresentasikan karakteristik konten dari setiap judul. Genre dianggap relevan dan informatif dalam menentukan kesesuaian atau kemiripan antar anime, sehingga dapat meningkatkan kualitas rekomendasi yang dihasilkan.

Beberapa tahapan untuk mengembangkan model ini:

1. TF-IDF Vectorizer

    Genre dari setiap anime diekstrak menggunakan teknik TF-IDF (Term Frequency–Inverse Document Frequency), yang mengubah teks genre menjadi representasi vektor numerik. Teknik ini menekankan genre yang unik atau jarang muncul sebagai informasi yang lebih penting dalam membedakan anime satu dengan lainnya

2. Consine Similarity

    Setelah genre diubah menjadi bentuk vektor, dilakukan perhitungan cosine similarity untuk mengukur tingkat kemiripan antar anime berdasarkan vektor TF-IDF mereka. Hasilnya adalah sebuah matriks simetris yang menunjukkan skor kemiripan antar setiap pasangan anime.

3. Top-N Recommendation

    Fungsi `anime_recommendation` dikembangkan untuk memberikan Top-N rekomendasi anime (secara default berjumlah 5 judul) yang memiliki kemiripan genre tertinggi terhadap anime yang dipilih pengguna. Output dari fungsi ini berupa daftar anime beserta informasi tambahan seperti nama anime, genre, dan tipe anime (TV, Movie, OVA, dll).

Kelebihan pendekatan ini:

- Personalized atau rekomendasi sangat relevan dengan preferensi pengguna karena didasarkan pada konten anime yang sebelumnya disukai.
- Pengguna dapat dengan mudah memahami alasan di balik rekomendasi karena penggunaan genre sebagai dasar terlihat polanya.
- Tidak memerlukan data pengguna lain karena sistem cukup mengetahui preferensi pengguna terhadap satu item.

Kekurangan pendekatan ini:

- Sistem cenderung merekomendasikan anime yang terlalu mirip dengan preferensi awal dan kurang mengeksplorasi genre lain yang mungkin juga disukai pengguna.
- Model tidak menggunakan informasi tambahan seperti rating, popularitas, durasi episode, atau ulasan pengguna yang bisa memperkaya rekomendasi.
- Sistem tidak belajar dari perilaku pengguna lain sehingga tidak dapat menemukan pola kesukaan yang lebih kompleks atau kolektif.

## Model Development - Collaborative Filtering

Model rekomendasi dikembangkan dengan arsitektur *RecommenderNet* yang mengadopsi konsep embedding untuk fitur pengguna dan anime. Arsitektur ini mencakup:

- User Embedding, yang mewakili karakteristik laten dari masing-masing pengguna.
- Anime Embedding, yang mewakili fitur laten dari tiap anime berdasarkan interaksi pengguna.
- Dot Product, yang digunakan untuk mengukur kesamaan atau kecocokan antara pengguna dan anime berdasarkan embedding mereka.
- Bias Terms, yang disertakan untuk mengoreksi preferensi spesifik dari pengguna dan popularitas anime tertentu.

Model dibangun menggunakan *TensorFlow* dan dikompilasi dengan fungsi loss *Binary Crossentropy* serta dioptimasi menggunakan *Adam optimizer*. Untuk meningkatkan performa pelatihan, digunakan teknik *Early Stopping* dengan monitoring terhadap `val_loss`.

**Top-N Recommendation Output**

Setelah model dilatih, sistem dapat menghasilkan Top 10 anime recommendation untuk pengguna tertentu berdasarkan prediksi skor interaksi tertinggi terhadap anime yang belum ditonton. 
Berikut adalah contoh output sistem untuk pengguna dengan ID 45:

Anime yang paling disukai pengguna:
- Ouran Koukou Host Club – Comedy, Romance, Shoujo
- Vampire Knight – Drama, Mystery, Romance
- Skip Beat! – Comedy, Drama, Romance
- Angel Beats! – Action, Comedy, Drama
- Kaichou wa Maid-sama! – Comedy, Romance, School

Top 10 Anime Rekomendasi:
- Monster – Psychological, Thriller
- Gintama – Action, Comedy, Parody
- Clannad: After Story – Drama, Romance, Slice of Life
- Fullmetal Alchemist: Brotherhood – Action, Drama, Fantasy
- Hunter x Hunter (2011) – Action, Adventure, Shounen
- Natsume Yuujinchou Shi – Supernatural, Slice of Life
- Gintama Movie: Kanketsu-hen – Action, Comedy
- Gintama': Enchousen – Comedy, Parody
- Kimi no Na wa. – Drama, Romance, Supernatural
- Haikyuu!!: Karasuno Koukou VS Shiratorizawa – Sports, Drama

Berdasarkan hasil di atas, terlihat bahwa sistem dapat memberikan rekomendasi yang cukup relevan dengan mempertimbangkan genre dan gaya anime yang sebelumnya disukai pengguna, seperti Shoujo, Romance, dan Supernatural, dan juga memberikan variasi seperti Action dan Psychological Thriller, yang berpotensi menarik bagi pengguna.

Kelebihan pendekatan ini:

- Model dapat menangkap hubungan kompleks antara pengguna dan item.
- Model dapat diskalakan ke jutaan pengguna dan item untuk menangani jutaan pengguna dan item secara efisien.
- Setiap pengguna dan item direpresentasikan secara unik dalam ruang vektor sehingga rekomendasi sangat disesuaikan dengan preferensi masing-masing pengguna.

Kekurangan pendekatan ini:

- Model tidak bisa memberikan rekomendasi yang baik untuk pengguna atau item baru yang belum memiliki interaksi apapun (zero interaction).
- Agar model dapat belajar representasi yang efektif, dibutuhkan dataset yang cukup besar dan memiliki kualitas data interaksi yang baik.
- Sulit untuk menjelaskan alasan spesifik di balik rekomendasi.


# Evaluation

## Evaluation Content-Based Filtering

Metrik yang digunakan untuk mengevaluasi kinerja sistem rekomendasi berbasis konten adalah Precision@N (Precision at Top-N). Metrik ini dipilih karena sangat relevan dengan tujuan sistem, yaitu memberikan daftar rekomendasi anime terbaik sebanyak N item yang sesuai dengan preferensi pengguna, dalam hal ini genre.

Precision@N mengukur proporsi item yang relevan (sesuai dengan preferensi/genre pengguna) terhadap jumlah total item yang direkomendasikan dalam daftar Top-N.
Rumusnya:

$$ Precision@N = (\frac{\text{Jumlah rekomendasi yang revelan}}{\text{Jumlah total item yang direkomendasikan (N)}}) * 100% $$

Metrik ini bekerja dengan menghitung berapa banyak dari rekomendasi yang benar-benar relevan (misalnya, memiliki genre yang sama atau sangat mirip dengan anime input) dari seluruh item yang direkomendasikan.

**Hasil Evaluasi**

Pengujian dilakukan terhadap anime "Haikyuu!!: Karasuno Koukou VS Shiratorizawa Gakuen Koukou", yang memiliki genre: *Comedy, Drama, School, Shounen, dan Sports*.

Selanjut dilakukan pemanggilan fungsi untuk menghasilkan 10 rekomendasi (Top 10) anime dengan genre serupa atau sangat mendekati melalui kode berikut: `anime_recommendation('Haikyuu!!: Karasuno Koukou VS Shiratorizawa Gakuen Koukou', k=10)`. 

Berikut hasil rekomendasi yang diberikan sistem:

| No. | Anime Name                                             | Genre                                                | Type   |
|-----|--------------------------------------------------------|------------------------------------------------------|--------|
| 1   | Haikyuu!! Movie 1: Owari to Hajimari                   | Comedy, Drama, School, Shounen, Sports               | Movie  |
| 2   | Slam Dunk                                              | Comedy, Drama, School, Shounen, Sports               | TV     |
| 3   | Haikyuu!! Movie 2: Shousha to Haisha                   | Comedy, Drama, School, Shounen, Sports               | Movie  |
| 4   | Rokudenashi Blues                                      | Comedy, Drama, School, Shounen, Sports               | Movie  |
| 5   | Haikyuu!! Second Season                                | Comedy, Drama, School, Shounen, Sports               | TV     |
| 6   | Batsu & Terry                                          | Comedy, Drama, School, Shounen, Sports               | Movie  |
| 7   | Haikyuu!!                                              | Comedy, Drama, School, Shounen, Sports               | TV     |
| 8   | Rokudenashi Blues 1993                                 | Action, Comedy, Drama, School, Shounen, Sports       | Movie  |
| 9   | Prince of Tennis: The National Tournament Finals       | Comedy, School, Shounen, Sports                      | OVA    |
| 10  | Kuroko no Basket                                       | Comedy, School, Shounen, Sports                      | TV     |

Dari hasil tersebut, anime yang direkomendasikan memiliki setidaknya 4 hingga 5 genre yang tumpang tindih dengan genre anime input. Oleh karena itu, semua item yang direkomendasikan dianggap relevan.

**Perhitungan Precision**

$$ Precision@N = (\frac{10}{10}) * 100\% = 100\% $$

Nilai Precision@10 sebesar 100% menunjukkan bahwa sistem berhasil memberikan rekomendasi yang sangat relevan dari sisi genre. Ini mencerminkan bahwa pendekatan content-based filtering yang digunakan mampu memahami dan mencocokkan karakteristik anime dengan sangat baik, khususnya dalam hal genre. Dengan demikian, sistem bekerja efektif dan akurat dalam memberikan saran yang sesuai dengan preferensi pengguna.

## Evaluation Collaborative Filtering

Metrik yang digunakan untuk mengevaluasi kinerja sistem rekomendasi berbasis kolaboratif adalah Root Mean Squared Error (RMSE). Metrik ini merupakan metrik yang umum digunakan untuk mengukur kesalahan prediksi dalam model rekomendasi berbasis regresi atau prediksi rating. RMSE memberikan gambaran seberapa jauh prediksi model menyimpang dari nilai aktual dalam satuan yang sama dengan data asli.

RMSE didefinisikan sebagai berikut:

$$
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

dengan:
- $y_i$ adalah nilai aktual dari data.
- $\hat{y}_i$ adalah nilai prediksi dari model.
- $n$ adalah jumlah total sampel dalam dataset.

RMSE mengkuadratkan kesalahan prediksi sehingga penalti terhadap kesalahan besar lebih berat, kemudian mengambil akar kuadrat agar satuan hasil tetap sama dengan data asli.


**Hasil Evaluasi RMSE**

![Result Evaluation RMSE](/images/projects/6/image.png)

Berdasarkan grafik pelatihan model yang menunjukkan metrik root_mean_squared_error pada data training dan testing, didapatkan bahwa nilai RMSE stabil pada kisaran:

- Training RMSE ≈ 0.167
- Validation RMSE ≈ 0.168

Grafik menunjukkan bahwa model mengalami peningkatan performa yang cepat hingga epoch ke-3, lalu melandai dan stabil. Hal ini menunjukkan bahwa model memiliki kemampuan prediksi yang baik dan stabil setelah beberapa epoch pelatihan tanpa mengalami overfitting yang signifikan.

# Kesimpulan

- Sistem rekomendasi berbasis konten dan kolaboratif mampu membantu pengguna menemukan tontonan anime yang relevan dengan preferensi pribadi mereka secara efisien. Dengan memproses data rating dan interaksi pengguna, model dapat memberikan prediksi rekomendasi yang akurat sehingga pengguna tidak perlu memilih secara manual dari ribuan judul anime.

- Dengan menggabungkan beberapa pendekatan rekomendasi, seperti collaborative filtering dan content-based filtering, sistem dapat menangkap berbagai aspek preferensi pengguna secara lebih holistik dan menghasilkan rekomendasi yang lebih relevan dan personal.