import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from model import NeuralNetwork


def read_csv(csv_path):
    data = pd.read_csv(csv_path)
    le = LabelEncoder()
    df1 = data.copy(deep=True)
    df1['Sex'] = le.fit_transform(df1['Sex'])
    df1['ChestPainType'] = le.fit_transform(df1['ChestPainType'])
    df1['RestingECG'] = le.fit_transform(df1['RestingECG'])
    df1['ExerciseAngina'] = le.fit_transform(df1['ExerciseAngina'])
    df1['ST_Slope'] = le.fit_transform(df1['ST_Slope'])
    standard = StandardScaler()
    df1['Oldpeak'] = standard.fit_transform(df1[['Oldpeak']])
    df1['Age'] = standard.fit_transform(df1[['Age']])
    df1['RestingBP'] = standard.fit_transform(df1[['RestingBP']])
    df1['Cholesterol'] = standard.fit_transform(df1[['Cholesterol']])
    df1['MaxHR'] = standard.fit_transform(df1[['MaxHR']])
    X = df1[df1.columns.drop(['HeartDisease'])].values
    Y = df1['HeartDisease'].values
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    return x_train, x_test, y_train, y_test


def main():
    csv_path = './heart.csv'
    x_train, x_test, y_train, y_test = read_csv(csv_path)
    model = NeuralNetwork(input_features=x_train.shape[1], num_classes=2)


if __name__ == '__main__':
    main()
