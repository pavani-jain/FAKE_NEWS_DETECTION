# Fake News Detection

## Introduction
In today's digital age, the rapid spread of misinformation and fake news has become a significant concern. With the sheer volume of news articles published daily, it's challenging to discern credible information from deceptive content. This project aims to tackle this issue by building a machine learning-based system capable of detecting fake news articles. By leveraging various machine learning models, we aim to classify news articles as "Fake" or "Real," thereby contributing to the broader effort of combating misinformation.

## Project Overview
This project involves the development and evaluation of multiple machine learning models to detect fake news. We utilize a dataset containing both fake and real news articles, preprocess the text data, and apply several algorithms to train models that can accurately classify news articles.

### Objectives
- To preprocess text data from news articles.
- To train and evaluate different machine learning models for fake news detection.
- To compare the performance of these models and identify the most effective one.
- To provide a user-friendly interface for manual testing of news articles.

## Dataset
The dataset used in this project is composed of two main files:
- `Fake.csv`: Contains news articles identified as fake.
- `True.csv`: Contains news articles identified as real.

Each dataset includes the following columns:
- `title`: The title of the news article.
- `text`: The full text of the news article.
- `subject`: The subject category of the article.
- `date`: The publication date of the article.

### Data Preprocessing
To prepare the data for modeling, the following preprocessing steps are applied:
- **Text Normalization**: Converting text to lowercase, removing extra spaces, special characters, URLs, and links.
- **Feature Engineering**: Creating a new feature `class`, where `0` indicates Fake News and `1` indicates Real News.

A custom function `wordopt()` is used for text cleaning and normalization.

## Models Implemented
The following machine learning models are implemented:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Gradient Boosting Classifier**
4. **Random Forest Classifier**

Each model is trained on a split of 75% training data and 25% test data. Performance metrics such as accuracy, precision, recall, and F1-score are calculated for evaluation.

## Results
The models were evaluated on the test set, and their accuracy scores are as follows:

| Model                        | Accuracy  |
|------------------------------|-----------|
| Logistic Regression           | 98.69%    |
| Decision Tree Classifier      | 99.66%    |
| Gradient Boosting Classifier  | 99.53%    |
| Random Forest Classifier      | 98.93%    |

### Best Model
The **Decision Tree Classifier** achieved the highest accuracy at 99.66%, making it the best-performing model in this project.

## Manual Testing
For manual testing, a function `manual_testing()` is provided, allowing users to input a news article and receive predictions from all four models. This feature enables real-time testing of news articles against the trained models.

## How to Run
### Prerequisites
Ensure you have the following libraries installed:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn
Steps to Run the Project
Clone the Repository:

bash
Copy code
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
Install Dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Jupyter Notebook or Python Script:

Open the Jupyter notebook Fake_News_Detection.ipynb to explore the code and results.
Alternatively, run the script fake_news_detection.py for training and testing.
Manual Testing:

Use the manual_testing() function to test any news article manually.
Future Enhancements
Implementing more advanced models, such as Neural Networks or Transformer-based models like BERT.
Experimenting with additional features like sentiment analysis and source credibility.
Applying the model to real-time news streams for live fake news detection.
Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

License
This project is licensed under the MIT License - see the LICENSE file for details.

markdown
Copy code

### Key Sections Explained:
- **Introduction**: Provides context and motivation for the project.
- **Project Overview**: Briefly explains the project’s goals and scope.
- **Dataset**: Describes the data used, its sources, and preprocessing steps.
- **Models Implemented**: Lists the machine learning models used, their purpose, and evaluation.
- **Results**: Summarizes the performance of the models.
- **How to Run**: Instructions for setting up and running the project.
- **Future Enhancements**: Suggestions for future work and improvements.

