import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import plotly.express as px
from moduleV2 import module

#Setup my Page
st.set_page_config(
    page_title="Students Degree Analysis",  # Title of the page
    page_icon="🎓",  # Graduation cap emoji
    layout="wide",
)
warnings.filterwarnings("ignore")
plt.style.use('ggplot')


# Load the dataset of students' lifestyle
students = pd.read_csv('./student_lifestyle_dataset.csv')

# Set page configuration for Streamlit
sb = st.sidebar

# Select mode (either 'Analyzing Mode' or 'Predict My GPA')
mode = sb.selectbox(
    'Enter the mode:',
    ['Choose The Mode:', 'Analyzing Mode', 'Predict My GPA']
)

# Change mode variable to shorter identifiers
if mode == 'Analyzing Mode':
    mode = 'a'
elif mode == 'Predict My GPA':
    mode = 'p'

# Analyzing Mode
if mode == 'a':
    # Function to generate figures for plotting
    def generate_fig(title='', xLabel='', yLabel=''):
        fig = plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.xlabel(xLabel, fontsize=20)
        plt.ylabel(yLabel, fontsize=20)
        return fig

    mode2 = st.sidebar
    aMode = mode2.selectbox(
        'Questions:',
        ['Choose Your Question!', 'Where do most students spend their time?', 
         'What About The GPA Of These Students?', 'What Is Students Study Style?',
         'What Is The Relation Between Sleep Hours And GPA?', 'What Is The Relation Between Social Hours And GPA?',
         'What Happens When You Study More?', 'Relation Between Socialized Hours and GPA and Stress Level?'], key=2
    )

    # Mapping the selected question to a key for easier referencing
    choose_dic = {
        'Choose Your Question!': 'N',
        'Where do most students spend their time?': 'q1',
        'What About The GPA Of These Students?': 'q2',
        'What Is Students Study Style?': 'q3',
        'What Is The Relation Between Sleep Hours And GPA?': 'q4',
        'What Is The Relation Between Social Hours And GPA?': 'q5',
        'What Happens When You Study More?': 'q6',
        'Relation Between Socialized Hours and GPA and Stress Level?': 'q7'
    }
    aMode = choose_dic[aMode]

    # Analysis for 'Where do most students spend their time?'
    if aMode == 'q1':
        s = "Where do most students spend their time? <br> <br>"
        st.markdown(f"<p style='margin : -38px 0px; font-size:50px; font-family :Georgia; color:red;'>{s}</p>", unsafe_allow_html=True)
        hours = []
        # Calculate average time spent on each activity
        for column in students.columns[1:6]:
            hours.append(students[column].mean().round(2))

        # Create pie chart for time distribution
        fig = plt.figure(figsize=(7, 7))
        plt.pie(hours, labels=students.columns[1:6], autopct='%1.1f%%')
        plt.title("Average time for the students")


        st.pyplot(fig)
        st.divider()

        # Display insight on the results
        s = '''<br> As we see that most students are spending their time on <b> Studying and Sleeping </b> <br>
        Which Makes Us Take a Positive Intuition About Them!<br>'''
        st.markdown(f"<p style='margin : -38px 0px; font-size:20px; font-family :Georgia; color:black;'>{s}</p>", unsafe_allow_html=True)

    # GPA distribution analysis
    elif aMode == 'q2':
        avg = students['GPA'].mean()
        max_ = students['GPA'].max()
        min_ = students['GPA'].min()
        c1, c2, c3 = st.columns((20, 20, 20))
        c1.metric("Max GPA", max_)
        c2.metric('Least GPA', min_)
        c3.metric('Average GPA', avg.__round__(1))

        st.divider()

        # Plot GPA distribution
        fig = generate_fig("Distribution Of Students' GPA", xLabel="GPA OF THE STUDENTS", yLabel="Number of Students")
        sns.histplot(data=students, x='GPA', kde=True, color='teal')

        st.pyplot(fig)
        s = '''<br> As we see that the distribution is <b> Normally Distributed </b> <br> And Most Students Get <b> 2.75 -> 3.5 GPA </b> <br>
        <br>'''
        st.markdown(f"<p style='margin : -38px 0px; font-size:20px; font-family :Georgia; color:black;'>{s}</p>", unsafe_allow_html=True)

    # Study style analysis
    elif aMode == 'q3':
        avg = students['Study_Hours_Per_Day'].mean()
        max_ = students['Study_Hours_Per_Day'].max()
        min_ = students['Study_Hours_Per_Day'].min()
        c1, c2, c3 = st.columns((20, 20, 20))
        c1.metric("Max Hour", max_)
        c2.metric('Least Hour', min_)
        c3.metric('Average Hour', avg.__round__(1))

        st.divider()
        # Plot study hours distribution
        fig = generate_fig(title="Distribution Of Students' Study Hours", xLabel="Students Study Hours Per Day", yLabel="Number of Students")
        sns.histplot(data=students, x='Study_Hours_Per_Day', kde=True)
        st.pyplot(fig)

        s = '''<br> As we see that the distribution is <b> Balanced </b> And Data <b> Is not Biased </b> <br>
        Which Makes Us Take a Positive Intuition About Them!<br>
        <b> But What If We Include The Stress Level? </b>'''
        st.markdown(f"<p style='margin : -38px 0px; font-size:20px; font-family :Georgia; color:black;'>{s}</p>", unsafe_allow_html=True)

        st.divider()
        # Plot study hours vs. stress level
        fig = generate_fig(title="Distribution Of Students' Study Hours (WithIn Stress Level)",xLabel="Students Study Hours Per Day", yLabel="Number of Students")
        sns.histplot(data=students, x='Study_Hours_Per_Day', kde=True, hue='Stress_Level')
        st.pyplot(fig)

        s = '''<br>The Figure Wants To Say That <b> "The More Hours You Study, The More You Get Stressed!" </b>'''
        st.markdown(f"<p style='margin : -38px 0px; font-size:20px; font-family :Georgia; color:black;'>{s}</p>", unsafe_allow_html=True)

    # Repeat similar analysis for other questions ('q4', 'q5', etc.)
    # Each question will display metrics and plots relevant to different features (e.g., sleep hours, social hours, etc.)
    # The process involves calculating means, maximum, and minimum values, followed by plotting distributions, and providing insights
    # Many of the sections involve using histograms and scatter plots to show relationships with GPA, stress level, and other factors


    elif aMode == 'q4':
        avg = students['Sleep_Hours_Per_Day'].mean()
        max_ = students['Sleep_Hours_Per_Day'].max()
        min_ = students['Sleep_Hours_Per_Day'].min()
        c1,c2,c3 = st.columns((20,20,20))
        c1.metric("Max Sleep Hour", max_)
        c2.metric('Least Sleep Hour', min_)
        c3.metric('Average Sleep Hour', avg.__round__(1))

        st.divider()

        fig = generate_fig(xLabel="Students Sleep Hours Per Day", yLabel="Students Number")
        sns.histplot(data = students, x = 'Sleep_Hours_Per_Day', kde = True)
        st.pyplot(fig)
        s = '''<br>   As we see that the distrubuation is <b> Balanced </b> And Data <b> Is not Biased </b> <br>
        And The Average Sleep Hour Is 7.5 Which Is <b> Approximatly Healthy! </b>'''
        st.markdown(f"<p style='margin : -38px 0px; font-size:20px; font-family :Georgia; color:black; '>{s}</p>", unsafe_allow_html=True)

        st.divider()
        fig = generate_fig(title="The Relation Between Sleep Hours And Stress Level", xLabel="Students Sleep Hours Per Day", yLabel="Students Number")
        sns.histplot(data = students, x = 'Sleep_Hours_Per_Day', kde = True, hue='Stress_Level', palette='coolwarm', multiple='stack')
        st.pyplot(fig)

        s = '''<br>As The Figure Shown That <b> "The Least Hour You Sleep The More You Get UnderStress!" </b>
        <br> As We See Also That <b> The Stress Level Is Moderated IN [6-8] Hours </b>'''
        st.markdown(f"<p style='margin : -38px 0px; font-size:20px; font-family :Georgia; color:black; '>{s}</p>", unsafe_allow_html=True)
    
    elif aMode == 'q5':
        avg = students['Social_Hours_Per_Day'].mean()
        max_ = students['Social_Hours_Per_Day'].max()
        min_ = students['Social_Hours_Per_Day'].min()
        c1,c2,c3 = st.columns((20,20,20))
        c1.metric("Max Social Hour", max_)
        c2.metric('Least Social Hour', min_)
        c3.metric('Average Social Hour', avg.__round__(1))

        st.divider()
        fig = generate_fig(xLabel="Students Social Hours Per Day", yLabel="Students Number")
        sns.histplot(data = students, x = 'Social_Hours_Per_Day', kde = True, multiple='stack')

        st.pyplot(fig)
        s = '''<br>   As we see that the distrubuation is <b> Not Balanced </b> And Nearly Data <b> Is Left-Skewwed  </b> <br>
        Which Mean That Most Student <b> Absolutly Spend From 0 To 2 Socialized Hours </b>'''
        st.markdown(f"<p style='margin : -38px 0px; font-size:20px; font-family :Georgia; color:black; '>{s}</p>", unsafe_allow_html=True)

        st.divider()
        fig = generate_fig("The Relation Between Social Hours And Stress Level", "Students Social Hours Per Day", "Students Number")
        sns.histplot(data = students, x = 'Social_Hours_Per_Day', kde = True, hue='Stress_Level', palette='coolwarm', multiple='stack')
        st.pyplot(fig)

        s = '''<br>As The Figure Shown That <b> "The more you sit on social media the less resposiblites you have the less pressure on you!" </b>
        <br> As We See Also That <b> The Stress Level Is Moderated IN [6-8] Hours </b>'''
        st.markdown(f"<p style='margin : -38px 0px; font-size:20px; font-family :Georgia; color:black; '>{s}</p>", unsafe_allow_html=True)
        st.divider()
        
 
    elif aMode == 'q6':
        avg = students['Study_Hours_Per_Day'].mean()
        max_ = students['Study_Hours_Per_Day'].max()
        min_ = students['Study_Hours_Per_Day'].min()
        c1, c2, c3 = st.columns((20, 20, 20))
        c1.metric("Max Study Hour", max_)
        c2.metric('Least Study Hour', min_)
        c3.metric('Average Study Hour', avg.__round__(1))
        st.divider()

        fig = generate_fig("Relation Between Physical Activity Hours and Study Hours Per Day", 'Study Hours Per Day', 'Physical Activity Hours')
        sns.regplot(students, x='Study_Hours_Per_Day', y='Physical_Activity_Hours_Per_Day', scatter_kws={'alpha': 0.3}, line_kws={'color': 'blue'}, color='red')
        st.pyplot(fig)
        st.divider()

        fig = generate_fig("Relation Between Social Media Hours and Study Hours Per Day", 'Study Hours Per Day', 'Physical Activity Hours')
        sns.regplot(students, x='Study_Hours_Per_Day', y='Social_Hours_Per_Day', scatter_kws={'alpha': 0.3}, line_kws={'color': 'blue'}, color='red')
        st.pyplot(fig)
        st.divider()

        fig = generate_fig("Relation Between GPA and Study Hours Per Day", 'Study Hours Per Day', 'GPA')
        sns.regplot(students, x='Study_Hours_Per_Day', y='GPA', scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'}, color='green')
        st.pyplot(fig)
        st.divider()
        s = '''<br>From All Of These Figures We Can Know That :  <br> - <br> - '''
        st.markdown(f"<p style='margin : -38px 0px; font-size:20px; font-family :Georgia; color:black; '>{s}</p>", unsafe_allow_html=True)

        s = '''<br><ol>
            <li>The More You Study The More Your GPA Is!</li>
            <li>The More You Study The Least You Do Physical Activity</li>
            <li>The More You Study The Least You Waste Social Media Hours!</li>
            <li>The More You Sit In SM The Less You Get GPA</li>
            </ol> <br> '''
        st.markdown(f"<p style='margin : -38px 0px; font-size:20px; font-family :Georgia; color:black; '>{s}</p>", unsafe_allow_html=True)
    elif aMode == 'q7':
            bins1 = np.linspace(students['Social_Hours_Per_Day'].min(), students['Social_Hours_Per_Day'].max(), 4)
            labels1 = ['Low', 'Mideum', 'Addict']
            SMlevels = pd.cut(students['Social_Hours_Per_Day'], bins = bins1, labels = labels1)
            students['SMlevels'] = SMlevels

            fig = generate_fig("The Relation Between Social Addict Levels And Stress Level", "Students Social Addict Levels", "Students Number")

            sns.histplot(students, x='SMlevels', hue = 'Stress_Level', palette='coolwarm', multiple='stack')
            st.pyplot(fig)  
            st.divider()

            st.write('Relation Between Social Media Levels and Average GPA')
            df = students.groupby('SMlevels')['GPA'].agg('mean').to_frame().style.background_gradient(cmap='OrRd')
            
            st.table(df)
            st.divider()

            st.write('Relation Between Social Media Levels and Stress Percantage')
            series = students.groupby(['SMlevels','Stress_Level'])['Student_ID'].count() / 20
            df = series.to_frame().rename({'Student_ID': 'Students Percantage'}, axis = 1).reset_index()
            df = df.style.background_gradient(cmap='RdYlGn')

            st.table(df)
            st.divider()
            
            fig = generate_fig('Linear Relation Between Socialized Hours Numbers and GPA', 'Socialized Hours Numbers', 'GPA')
            sns.regplot(students, x = 'Social_Hours_Per_Day', y='GPA', scatter_kws={'alpha':0.3}, line_kws = {'color':'orange'}, color='green')
            st.pyplot(fig)

            st.divider()
            
            fig = generate_fig('Stress Level and Socialized Hours Numbers and GPA In one Picture', 'Socialized Hours Numbers', 'GPA')
            sns.scatterplot(students, x = 'Social_Hours_Per_Day', y='GPA', hue='Stress_Level', palette='coolwarm')        
            st.pyplot(fig)  

            s = '''<br>From All Of These Figuers We Can Know That :  <br> - <br> - '''
            st.markdown(f"<p style='margin : -38px 0px; font-size:20px; font-family :Georgia; color:black; '>{s}</p>", unsafe_allow_html=True)
            s = '''<br><ol>
                <li>The More You Engage In Studying The More You Be UnderPressure The Less You Sit In SM.</li>
                <li>The More You Study The More Your GPA Is!</li>
                <li>The More You Don't Study The Less You UnderPressure The More You Sit In SM</li>
                <li>The More You Sit In SM The Less You GET GPA</li>
                </ol> <br> '''
            st.markdown(f"<p style='margin : -38px 0px; font-size:20px; font-family :Georgia; color:black; '>{s}</p>", unsafe_allow_html=True)
            st.divider()  
    else:
        s = "Welcome To 'Student Life Style Analyzing' Page! <br> <br>"
        st.markdown(f"<p style='margin : -38px 0px; font-size:50px; font-family :Encode Sans; color:red; '>{s}</p>", unsafe_allow_html=True)
        s = '''
        I follow Question and Answer Techinque!
        Choose Your Qeustion From The SideBar, And You Will Find Total Analyzing About It!
        To Start, Choose The Question From The SideBar Left!'''
        st.markdown(f"<p style=' font-size:25px; font-family :Encode Sans; color:black; '>{s}</p>", unsafe_allow_html=True)
        st.divider()
        st.divider()

        s1 = 'About The Data:'
        s2 = ' Data is Mainly Focus On : '
        s = '''
            The Relation Between Students LifeStyle And Their GPA! <br><br>
            '''
        st.markdown(f"<p style=' font-size:25px; font-family :Encode Sans; color:red; '>{s1}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style=' font-size:25px; font-family :Encode Sans; color:red; '>{s2}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style=' font-size:25px; font-family :Encode Sans; color:black; '>{s}</p>", unsafe_allow_html=True)
        st.markdown('[Data Link](https://www.kaggle.com/datasets/steve1215rogg/student-lifestyle-dataset)')

    
elif mode == 'p':
    # Check if the model has been fitted before
    if 'Fitted' not in st.session_state:
        st.session_state['Fitted'] = False
    # Welcome message for the 'Predict My GPA' page
    s = "Welcome To 'Predict My GPA Page' Page! <br> <br>"
    st.markdown(f"<p style='margin : -38px 0px; font-size:50px; font-family :Encode Sans; color:red; '>{s}</p>", unsafe_allow_html=True)
    
    # Fit the model if it's not already fitted
    if 'Fitted' in st.session_state and (not st.session_state['Fitted']):
        df = pd.read_csv('./student_lifestyle_dataset.csv')
        y = df['GPA'].values[:1999]
        df.drop(index=1, columns=['Stress_Level','GPA','Student_ID'], inplace=True)
        X = df.values
        myMod = module(X, y)
        myMod.fit()
        st.session_state['myMod'] = myMod
        st.session_state['Fitted'] = True

    # Input fields for the user to provide their data
    avgST = st.number_input('Enter Your Average Sleep Time In Hours: ', min_value=3, max_value=10)
    avgSoT = st.number_input('Enter Your Average Socialized Time In Hours: ', min_value=0, max_value=6)
    avgStT = st.number_input('Enter Your Average Study Time In Hours: ', min_value=1, max_value=8)
    avgPT = st.number_input('Enter Your Average Physical Time In Hours: ', min_value=0, max_value=5)
    avgET = st.number_input('Enter Your Average Extracurricular Time In Hours: ', min_value=0, max_value=6)
    
    # Prepare the input sample and predict GPA
    sample = [avgStT, avgET, avgST, avgSoT, avgPT]  # Study, Extracurricular, Sleep, Social, Physical
    def predict(sample):
        myMod = st.session_state['myMod']
        p = myMod.predict(sample)
        c1,c2,c3 = st.columns((20, 20, 20))
        c1.metric("Your GPA", p)
        c3.metric("Model Accuracy", myMod.performance())
        st.divider()
    
    st.button('Predict GPA!', on_click=predict, kwargs={'sample': sample})

    st.divider()

    # Notes about the data and the model
    s = '''<br> Important Notes: -The Data was originally collected via <b>A Google Form Survey!</b> <br> -I built the model from scratch! -Model Accuracy is 92.07%<br>'''
    st.markdown(f"<p style='margin : -38px 0px; font-size:20px; font-family :Georgia; color:black; '>{s}</p>", unsafe_allow_html=True)

else:
    # Default welcome message and information for other pages
    sb.markdown("Made with [Eng/Mohamed Saad](https://www.linkedin.com/in/ibnsa3d/):heart_eyes:")
    s = "Welcome To 'Predict Your GPA' Website!"
    st.markdown(f"<p style='margin : -38px 0px; font-size:50px; font-family :Encode Sans; color:red; '>{s}</p>", unsafe_allow_html=True)
    s = 'To start, choose the mode from the sidebar left!'
    st.markdown(f"<p style='font-size:20px; font-family :Encode Sans; color:black; '>{s}</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown('''
                This Website was totally developed by Eng. Mohamed Saad
                ''')
   
