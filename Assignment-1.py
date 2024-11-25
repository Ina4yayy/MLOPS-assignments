#Name : Indrayani Gaidhane            Roll no : 19
#Batch : A1

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
sns.set()
plt.style.use('ggplot')


df= pd.read_csv('gender_submission.csv')
df.head()
df['Survived'].value_counts()
df.describe()
df = pd.read_csv("test.csv")
df.head()

class TitanicEDA():
    def __init__(self,data_path):
        self.data=self.load_data(data_path)
    def load_data(self,data_path):
        return pd.read_csv(data_path)

    def statistics(self):
        return self.data.describe()

    def statistics2(self):
        print("")
        return self.data.describe(include=['object'])

    def statistics3(self):
        print("")
        data_summary=pd.DataFrame(self.data.dtypes)
        data_summary['Missing_Values']=self.data.isnull().sum()
        data_summary['Nunique']=self.data.nunique()
        data_summary['Count']=self.data.count()
        data_summary=data_summary.rename(columns={0:'Data_Type'})
        data_summary
        return data_summary
      
    def visualization(self,feature,save_path):
        plt.figure(figsize=(10,6)) 
        group=self.data.groupby(feature)['Survived'].mean()
        group.plot(kind="bar")
        plt.title(f'Survival rate by {feature}')
        plt.xlabel(feature)
        plt.ylabel('Survival Rate')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close()

    def visualization_of_age(self,save_path):
        plt.figure(figsize=(10,6))
        survived=self.data[self.data['Survived']==1]['Age']
        not_survived=self.data[self.data['Survived']==0]['Age']
        plt.hist([survived,not_survived],label=['Survived','Not_Survived'],stacked=True)
        plt.title("Age Distribution by Survival")
        plt.xlabel('Age')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close()

    def implement(self):
        print(self.statistics())
        print(self.statistics2())
        print(self.statistics3())
        self.visualization('Pclass','survival_class.png')
        self.visualization('Sex','survival_gender.png')
        self.visualization('Embarked','survival_embarked.png')
        self.visualization('SibSp','survival_sibsp.png')
        self.visualization('Parch','survival_parch.png')
        self.visualization_of_age('Age.png')
     

result=TitanicEDA('train.csv')
result.implement()
