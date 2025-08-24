# 🎬 Hybrid Movie Recommender System

This project is an interactive movie recommender system built with Streamlit, developed as part of my **#30DaysOfAI** challenge. The application utilizes a hybrid model to provide personalized movie suggestions based on both content similarity and collaborative filtering.

## ✨ Key Features

-   **Hybrid Recommendation Engine:**
    -   **Content-Based Filtering:** Recommends movies by comparing their textual features like plot, genres, keywords, and director using TF-IDF and Cosine Similarity.
    -   **Collaborative Filtering:** Suggests movies based on the tastes of similar users, powered by the SVD algorithm.
-   **Interactive UI:** A user-friendly and modern web interface built with Streamlit.
-   **Multi-language Support:** The interface is available in both English and Turkish.

## 🚀 Live Demo

You can try the live version of the application here: **[INSERT YOUR STREAMLIT CLOUD LINK HERE]**

## 📸 Screenshot

![Application Screenshot](https://i.imgur.com/your_screenshot_image.png)

*(You should take a screenshot of your running app, upload it to a site like Imgur, and update this link.)*

## 💻 Tech Stack

-   **Python**
-   **Streamlit:** For the interactive web interface
-   **Pandas:** For data manipulation and analysis
-   **Scikit-learn:** For TF-IDF vectorization
-   **Surprise:** A scikit for building recommender systems (used for SVD)
-   **NumPy:** For numerical and vector operations

## ⚙️ Setup and Usage

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SuleymanToklu/Day12-Movie-Recommender-System.git](https://github.com/SuleymanToklu/Day12-Movie-Recommender-System.git)
    cd Day12-Movie-Recommender-System
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

The application will be available at `http://localhost:8501`. The pre-trained model files (`.pkl`) are included in the repository, so no training is required to run the app.
