import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px



df=pd.read_csv("all_Rt_Lt_for_dataVis.csv")
df_w=df[:174] #for widrh only use first 174 rows as the following 174 rows (right sude) are for left sside and width measuremnts do not have left or right (they are same values for right and left)

#two tabs 1 visualization 1 classifier

# Tabbed layout
tab1, tab2 = st.tabs(["Graphs", "Occlusion Classifier"])

with tab1:
    st.title("Growth and Development of Human Dentition")

#st.write("This is an app to display data and charts regarding dental development.")
#
#tab2: st.write("Many kids with Class II occlusion transition to Class I occlusion, which is thought to be more ideal, with growth and development. However, some of them still remain Class II occlusion in adult dentition. This app provides possibility of maintained Class II occlusion in adult dentition")


    # Sidebar
    st.sidebar.markdown("Use the filters below to explore data.")

    # Age filter
    min_age = int(df_w["Age"].min()) #had error - asked Gen AI and found that need int()
    max_age = int(df_w["Age"].max())

    age_range = st.sidebar.slider(
        "Select age range:",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age),
        step=1
    )
    
   #Initially used slider to select an age, but later, wanted to make it a range, I tried to change it to a range but did not work, got help from Gen AI: I want to select age range than an age?

    # sex filter
    sex= st.sidebar.multiselect("Select sex:", options=df["sex"].unique(), default=df["sex"].unique())

    # State filter
    states = st.sidebar.multiselect("Select Growth collection states:", options=df_w["set_states"].unique(), default=["Michigan", "Oregon"])


    # Filter the data
    filtered_df = df[
        (df["Age"] >= age_range[0]) &
        (df["Age"] <= age_range[1]) & (df["set_states"].isin(states))&
        (df["sex"].isin(sex))]
    
    filtered_df_w = df_w[
        (df_w["Age"] >= age_range[0]) &
        (df_w["Age"] <= age_range[1]) & (df_w["set_states"].isin(states))&
        (df_w["sex"].isin(sex))]

 
    st.subheader("Maxillary Inter Canine Width vs Age")
    fig1 = px.scatter(filtered_df_w, x="Age", y="UC-C / U3-3", color="sex",
        hover_name="set_states", #want to control min and max age using slider
        )
    st.plotly_chart(fig1, use_container_width=True)

    
    st.subheader("Mandibular Inter Canine Width  vs Age")
    fig2 = px.scatter(filtered_df_w, x="Age", y="LC-C / L3-3", color="sex",
        hover_name="set_states", #want to control min and max age using slider
        )
    st.plotly_chart(fig2, use_container_width=True)
    st.subheader("Class vs Timepoint")
        
    df_filtered = df[df['Timepoint'].isin(['T2', 'T3'])]
    class_counts = df_filtered.groupby(['Timepoint', 'class_cat']).size().reset_index(name='Count')


    st.subheader("Classification vs Timepoint")
        
    df_filtered = df[df['Timepoint'].isin(['T2', 'T3'])]
    class_counts = df_filtered.groupby(['Timepoint', 'class_cat']).size().reset_index(name='Count')
    
        
    fig3 = px.bar(
    class_counts,
    x='class_cat',
    y='Count',
    color='Timepoint',
    barmode='group',
    labels={'class_cat': 'Classification', 'Count': 'Number of Subjects'},
    #title="Classification vs Timepoint (Grouped)"
    )

    st.plotly_chart(fig3, use_container_width=True)


    df_filtered['Subject_Side'] = df_filtered['Subject_ID'].astype(str) + "_" + df_filtered['Side']
    
    fig4 = px.line(
    df_filtered,
    x='Timepoint',
    y='class_cat',
    color='Subject_Side',
    markers=True,
    labels={'class_cat': 'Classification', 'Timepoint': 'Timepoint'},
    )

    st.subheader("Classification Change Over Time per Subject on Each Side")
    st.plotly_chart(fig4, use_container_width=True)


        
## Show results
#st.write(f"Showing data for {sex} and {age} in {', '.join(states)}")
#st.dataframe(filtered_df)

# Expandable section
with st.expander("See raw data"):
    st.dataframe(filtered_df)



with tab2:
    st.title("Class II Occlusion Prediciton")
    st.write("Many kids with Class II occlusion transition to Class I occlusion, which is thought to be more ideal, with growth and development. However, some of them still remain Class II occlusion in adult dentition. This app provides possibility of maintained Class II occlusion in adult dentition")

    #We wanted to have a prediction (classifier) based on logistic regression coefficients - had coefficients already from other research project.
    #To present possibility values based on the coefficients and inout values, we needed help from Gen AI
    
    # Add interactive text input
    #Input
    st.sidebar.markdown("Use fields below to input measurements for Occlusion Classifier")
    
    overjet=st.sidebar.number_input("Overjet (mm)", value=3.0)
    postant=st.sidebar.number_input("Post/Ant Ratio", value=0.65)
    u66=st.sidebar.number_input("U6-6 Width (mm)", value=38.5)
    artpog=st.sidebar.number_input("Art-Pog (mm)", value=85.0)
    uleeway=st.sidebar.number_input("Leeway Space (Mx, mm)", value=2.7)
    classcat=st.sidebar.selectbox("Molar Classification", options=[("Class I",0), ("Class II",1)], index=1) #did number_input initially did not work - aksed Gen AI: why error?

    # Coefficients and Stats from an earlier project with the same dataset
    coefs = {
        "const": -1.0573,
        "Overjet(11)": 1.2704,
        "post/ant": -0.7261,
        "U6-6_cusptip": -0.8129,
        "art_pog": -0.3036,
        "U_leeway": -0.3844,
        "class_cat_T2": 0.9187
    }

    means = {
        "Overjet(11)": 3.176000,
        "post/ant": 0.654154,
        "U6-6_cusptip": 38.553571,
        "art_pog": 85.498626,
        "U_leeway": 2.742571
    }

    stds = {
        "Overjet(11)": 1.576033,
        "post/ant": 0.035208,
        "U6-6_cusptip": 3.172536,
        "art_pog": 4.377117,
        "U_leeway": 1.263946
    }

    #Prediction
    raw= {
        "Overjet(11)": overjet,
        "post/ant": postant,
        "U6-6_cusptip": u66,
        "art_pog": artpog,
        "U_leeway": uleeway,
        "class_cat_T2": classcat[1]
    }
  #did not work, needed to use scaled values (logistic regression used standard scaler)

    numeric_df = df.select_dtypes(include='number')     #had error message bc did not have this line initially, asked Gen AI for help: why error?
    mean = numeric_df.mean().to_dict()
    stds = numeric_df.std().to_dict()



  #needed to use scaled values (logistic regression used standard scaler)
    scaled = {
        k: (v-mean[k])/stds[k] if k in stds else v
        for k,v in raw.items()
    }

    logit = coefs["const"] + sum(scaled[k] * coefs[k] for k in scaled)
    prob = 1 / (1 + np.exp(-logit))

    st.subheader("Predicted Probability of Class II at T3:")
    st.metric(label="Probability", value=f"{prob*100:.2f}%")

