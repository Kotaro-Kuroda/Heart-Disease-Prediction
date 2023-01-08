import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def read_csv(csv_path):
    data = pd.read_csv(csv_path)
    le = LabelEncoder()
    df1 = data.copy(deep = True)

    df1['Sex'] = le.fit_transform(df1['Sex'])
    df1['ChestPainType'] = le.fit_transform(df1['ChestPainType'])
    df1['RestingECG'] = le.fit_transform(df1['RestingECG'])
    df1['ExerciseAngina'] = le.fit_transform(df1['ExerciseAngina'])
    df1['ST_Slope'] = le.fit_transform(df1['ST_Slope'])
    
    standard= StandardScaler()
    df1['Oldpeak']= standard.fit_transform(df1[['Oldpeak']])
    df1['Age']= standard.fit_transform(df1[['Age']])
    df1['RestingBP']= standard.fit_transform(df1[['RestingBP']])
    df1['Cholesterol']= standard.fit_transform(df1[['Cholesterol']])
    df1['MaxHR']= standard.fit_transform(df1[['MaxHR']])
    return df1

def main():
    csv_path = './heart.csv'
    data = read_csv(csv_path)
    print(data['HeartDisease'])

if __name__ == '__main__':
    main()