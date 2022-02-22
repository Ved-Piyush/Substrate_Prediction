# load all the librries
import plotly.graph_objects as go
import dash
import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
import plotly.express as px
import numpy as np
from dash.dependencies import Input, Output, State
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
# seaborn that helps with aesthetically pleasing plots
import seaborn as sns
import matplotlib.pyplot as plt
import gensim
import dash_bootstrap_components as dbc
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold


model1 = gensim.models.doc2vec.Doc2Vec.load(r"D:\Gene_Project\Refined_Codes\doc2vec_on_unsupervised.model")

data = pd.read_csv(r"D://Gene_Project//pul_seq_low_high_substr_year_corrected.csv")
updated_data_supervised = pd.read_csv(r"D://Gene_Project//all_unsupervised_genes.csv")

# dom_eigen_val = pd.read_csv(r"D://Gene_Project//dominant_eigen.csv")

skewness_data = pd.read_csv(r"D://Gene_Project//skewness_data.csv")
combo_data = pd.read_csv(r"D://Gene_Project//accuracy.csv")
# combo_data = combo_data.merge(skewness_data, on = "high_level_substr")


# get the frequency counts
D = data.high_level_substr.value_counts()
# convert to a dictionary
D = dict(D)

freq_data = pd.DataFrame({"high_level_substr":D.keys(), "frequency":D.values() })

fig = px.bar(x = D.keys(), y = list(D.values()), title = "Frequencies for the high level substrates", 
             labels=dict(x="High Level Substrates", y="Frequency"))
# fig2 = px.bar(x = catch_df["high_level_substr"], y = catch_df["purity"])


data["length"] = [len(seq.replace("|",",").split(",")) for seq in data["sig_gene_seq"]]

D = dict(data.groupby("high_level_substr")["length"].std().reindex(D.keys()))

length_data = pd.DataFrame({"high_level_substr":D.keys(), "std_length":D.values() })



fig3 = px.bar(x = D.keys(), y = list(D.values()), title = "Standard Deviation for lengths of the gene sequences", 
              labels=dict(x="High Level Substrates", y="Standard Deviation"))

D = dict(data.groupby("high_level_substr")["low_level_substr"].nunique().reindex(D.keys()))

low_unique_data = pd.DataFrame({"high_level_substr":D.keys(), "num_unique_low_level_substr":D.values() })

# all_combined = pd.merge(freq_data, length_data, how = "inner", on = "high_level_substr").merge(low_unique_data, how = "inner", on = "high_level_substr")
all_combined = combo_data.copy().merge(freq_data, on = "high_level_substr")


fig4 = px.bar(x = D.keys(), y = list(D.values()), title = "Number of unique low level substrates", 
              labels=dict(x="High Level Substrates", y="Number of Unique"))


# fig5 = px.bar(x = dom_eigen_val["high_level_substr"], y = dom_eigen_val["explained_var"],
#               title = "Percentage variation in gene composition explained by the dominant eigenvector")


fig6 = px.bar(x = skewness_data["high_level_substr"], y = skewness_data["skewness"],
              title = "Skewness coefficient for the distance distribution", 
              labels=dict(x="High Level Substrates", y="Skewness coefficient"))


fig5 = px.bar(x = combo_data["high_level_substr"], y = combo_data["5-fold-average-f1"],
              title = "5 fold average F1 Scores", labels=dict(x="High Level Substrates", y="F1 Scores"))


three_d_data = pd.read_csv(r"D://Gene_Project//tsne_3d.csv")

x,y,z = three_d_data["0"], three_d_data["1"], three_d_data["2"]


fig_3_d = px.scatter_3d(three_d_data, x="0", y="1", z="2", color = "high_level_substr",
#                     color_discrete_map={'pectin':'red' , 'beta-mannan':'darkgreen' , 'galactan':'yellow' , 
#                                         'chitin':'blue', 'beta-glucan':'blueviolet', 'other':'brown', 
#                                         'starch':'coral', 'cellulose':'darkcyan', 'fructan':'hotpink', 
#                                         'cellooligosaccharide':'indigo', 'alpha-mannan':'lightseagreen', 'rare':'fuchsia' }, 
                   category_orders={"high_level_substr":data["high_level_substr"].value_counts().keys().tolist()}, 
                   width = 200, 
                   title = "T-SNE plot for High Level Substrates using the distance matrix based on gene composition")

fig_3_d.update_layout(scene = dict(
                    xaxis_title='Dimension 1',
                    yaxis_title='Dimension 2',
                    zaxis_title='Dimension 3'),
                   )


dropdown = []

for substr in data["high_level_substr"].value_counts().keys().tolist():
    dict1 = {"label": substr, "value": substr}
    dropdown.append(dict1)

dropdown.append({"label":"No Classes selected", "value":''})


method_dropdown = [{"label":"Single Stage Model", "value": "Method1"}, 
                   {"label":"Two Stage Model", "value": "Method2"}]

ml_dropdown = [{"label":"Random Forest", "value": "RF"}, 
                   {"label":"Balanced Random Forest", "value": "BRF"}]
   
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

unsup_seq = []
for seq in updated_data_supervised["sequence"]: 
    dict1 = {"label": seq, "value":seq}
    unsup_seq.append(dict1)
    


unsup_genes = []
for seq in model1.wv.index_to_key: 
    dict1 = {"label": seq, "value":seq}
    unsup_genes.append(dict1)    
    

    
app = dash.Dash( external_stylesheets=external_stylesheets)
server = app.server
    
app.layout = html.Div([
        dcc.Tabs([
            dcc.Tab(label='Exploratory Plots for High Level Substrates', children=[
                dcc.Graph(
                    figure=fig, 
                    style={
                           
                            "border": "3px #5c5c5c solid"
                           
                        }
                ) ,
                
                
                
                dcc.Graph(
                    figure=fig3, 
                    style={
                           
                            "border": "3px #5c5c5c solid"
                           
                        }
               ) ,
                
                dcc.Graph(
                    figure=fig4, 
                    style={
                           
                            "border": "3px #5c5c5c solid"
                           
                        }
               ) ,
                
                
                dcc.Graph(
                    figure=fig6, 
                    style={
                           
                            "border": "3px #5c5c5c solid"
                           
                        }
               ) ,
                
                
                dcc.Graph(
                    figure=fig5, 
                    style={
                           
                            "border": "3px #5c5c5c solid"
                           
                        }
               ) ,
                
                
                
                
                
                dcc.Graph(
                    figure= fig_3_d,
                    style={
                            "width": "100%",
                            "height": "800px",
                            "display": "inline-block",
                            "border": "3px #5c5c5c solid",
                            "padding-top": "5px",
                            "padding-left": "1px",
                            "overflow": "hidden"
                        }
                    
                ) 
                
            ]), 
            
    dcc.Tab(label = "Machine Learning & Evaluation", 
            children = [
                
                 html.Div( ## Select menus
            [
                html.Div([
                    
                    dcc.Dropdown(
                            id = "to_keep_classes",
                            options=dropdown,
                            multi=True,
                            searchable = True, 
                            placeholder = "High Level Substrates", 
                            value = ''
                                )
                    
                    ]
                    ,
                    style={
                        "display": "inline-block",
                        "width": "15%",
                        "verticalAlign": "top"
                    }),
                
                
                html.Div([
                    
                    dcc.Input(
                            id = "frequency_target", 
                            type = "number", 
                            placeholder = "Frequency greater than"
                            # value = ["multiple_substrates", "xylan", "pectin"]
                                )
                    
                    ],
                    style={
                        "display": "inline-block",
                        "width": "12%",
                        "margin-left": "2%",
                        "verticalAlign": "top"
                    }), 
                
                
                html.Div([
                    
                    dcc.Input(
                            id = "std_var_gene_seq_len", 
                            type = "number", 
                            placeholder = "Std lengths less than"
                            # value = ["multiple_substrates", "xylan", "pectin"]
                                )
                    
                    ],
                    style={
                        "display": "inline-block",
                        "width": "12%",
                        "margin-left": "5%",
                      
                        "verticalAlign": "top"
                    }), 
                
                
                html.Div([
                    
                    dcc.Input(
                            id = "number_unique_low_level", 
                            type = "number", 
                            placeholder = "# unique less than"
                            # value = ["multiple_substrates", "xylan", "pectin"]
                                )
                    
                    ],
                    style={
                        "display": "inline-block",
                        "width": "12%",
                        "margin-left": "4%",
                        "verticalAlign": "top"
                    }), 
                
                
                html.Div([
                    
                    dcc.Input(
                            id = "skewness_coef", 
                            type = "number", 
                            placeholder = "Skew coef greater than"
                            # value = ["multiple_substrates", "xylan", "pectin"]
                                )
                    
                    ],
                    style={
                        "display": "inline-block",
                        "width": "12%",
                        "margin-left": "4%",
                        "verticalAlign": "top"
                    }), 
                
                html.Div([
                    
                    dcc.Input(
                            id = "univariate_f1_score", 
                            type = "number", 
                            placeholder = "f1_score greater than"
                            # value = ["multiple_substrates", "xylan", "pectin"]
                                )
                    
                    ],
                    style={
                        "display": "inline-block",
                        "width": "17%", 
                        "margin-left": "4%",
                        "verticalAlign": "top"
                    }),
                
                html.Div([
                    
                    dcc.Dropdown(
                            id = "method", 
                            options=method_dropdown,
                            multi=False,
                            searchable = True,
                            placeholder = "Select ML Method"
                                )
                    
                    ],
                    style={
                        "display": "inline-block",
                        "width": "15%",
                        "verticalAlign": "top"
                    }), 
                
                html.Div([
                    
                    dcc.Dropdown(
                            id = "ml_method",
                            options=ml_dropdown,
                            multi=False,
                            searchable = True, 
                            placeholder = "Select ML Algo"
                                )
                    
                    ],
                    style={
                        "display": "inline-block",
                        "width": "13.5%", 
                        "margin-left": "2%",
                        "verticalAlign": "top"
                    }), 
                
                
                html.Div([
                    
                    dcc.Input(
                            id = "max_features", 
                            type = "number", 
                            placeholder = "number of gene tokens", 
                            max = len(np.unique([gene for seq in data["sig_gene_seq"] for gene in seq.replace("|", ",").split(",")]))
                            # value = ["multiple_substrates", "xylan", "pectin"]
                                )
                    
                    ],
                    style={
                        "display": "inline-block",
                        "width": "12%", 
                        "margin-left": "3.5%",
                        "verticalAlign": "top"
                    }),
                
                
                
                
                
                html.Div([
                    html.Button('GO', id='button')
                    ])
                
            ]
        ), 
                 html.Div([
                     
                     dcc.Graph(id = "feature-graphic", 
                               style={
                            "width": "100%",
                            "height": "800px",
                            "display": "inline-block",
                            "border": "3px #5c5c5c solid",
                            "padding-top": "5px",
                            "padding-left": "1px",
                            "overflow": "hidden"
                        })
                     
                     
                     ]), 
                 
                 html.Div([
                     
                     dcc.Graph(id = "feature-graphic1", 
                               style={
                            "width": "100%",
                            "height": "800px",
                            "display": "inline-block",
                            "border": "3px #5c5c5c solid",
                            "padding-top": "5px",
                            "padding-left": "1px",
                            "overflow": "hidden"
                        })
                     
                     
                     ]), 
                 
                 
                 html.Div([
                     
                     dcc.Graph(id = "feature-graphic2", 
                               style={
                            "width": "100%",
                            "height": "800px",
                            "display": "inline-block",
                            "border": "3px #5c5c5c solid",
                            "padding-top": "5px",
                            "padding-left": "1px",
                            "overflow": "hidden"
                        })
                     
                     
                     ]), 
                 
                 
                 html.Div([
                     
                     dcc.Graph(id = "feature-graphic3", 
                               style={
                            "width": "100%",
                            "height": "800px",
                            "display": "inline-block",
                            "border": "3px #5c5c5c solid",
                            "padding-top": "5px",
                            "padding-left": "1px",
                            "overflow": "hidden"
                        })
                     
                     
                     ]), 
                 
                 html.Div([
                     
                     dcc.Graph(id = "feature-graphic4", 
                               style={
                            "width": "100%",
                            "height": "800px",
                            "display": "inline-block",
                            "border": "3px #5c5c5c solid",
                            "padding-top": "5px",
                            "padding-left": "1px",
                            "overflow": "hidden"
                        })
                     
                     
                     ]), 
                 
                 html.Div([
                     
                     dcc.Graph(id = "feature-graphic5", 
                               style={
                            "width": "100%",
                            "height": "800px",
                            "display": "inline-block",
                            "border": "3px #5c5c5c solid",
                            "padding-top": "5px",
                            "padding-left": "1px",
                            "overflow": "hidden"
                        })
                     
                     
                     ])
                 
                 # html.Div([
                     
                 #     html.Div(id = "cm_multi", 
                 #               style={
                 #            "width": "100%",
                 #            "height": "300px",
                 #            "display": "inline-block",
                 #            "border": "3px #5c5c5c solid",
                 #            "padding-top": "5px",
                 #            "padding-left": "1px",
                 #            "overflow": "hidden"
                 #        })
                     
                     
                 #     ]), 
                 
                 # html.Div([
                     
                 #     html.Div(id = "cm_overall", 
                 #               style={
                 #            "width": "100%",
                 #            "height": "300px",
                 #            "display": "inline-block",
                 #            "border": "3px #5c5c5c solid",
                 #            "padding-top": "5px",
                 #            "padding-left": "1px",
                 #            "overflow": "hidden"
                 #        })
                     
                     
                 #     ])
                 
                 
                 
                
                
                ]), 
     dcc.Tab(label = "Search Engine for similar gene sequences and genes",
            children = [
                
                 html.Div( ## Select menus
            [
                html.Div([
                    
                    dcc.Input(
                            id = "gene_sequence", 
                            placeholder = "Enter gene sequence"
                            # value = ["multiple_substrates", "xylan", "pectin"]
                                )
                    
                    ],
                    style={
                        "display": "inline-block",
                        "width": "20%"
                    }),
                html.Div([
                    
                    dcc.Input(
                            id = "gene", 
                            placeholder = "Enter gene"
                                )
                    
                    ],
                    style={
                        "display": "inline-block",
                        "width": "30%",
                        "margin-left": "20px",
                        "verticalAlign": "top"
                    }), 
                

                
                html.Div([
                    html.Button('GO', id='button1')
                    ])
                
            ]
        ), 
                 html.Div([
                     
                     html.Div(id = "gene_seq_similar", 
                               style={
                            "width": "100%",
                            "height": "400px",
                            "display": "inline-block",
                            "border": "3px #5c5c5c solid",
                            "padding-top": "5px",
                            "padding-left": "1px",
                            "overflow": "hidden"
                        })
                     
                     
                     ]), 
                 
                 html.Div([
                     
                     html.Div(id = "gene_similar", 
                               style={
                            "width": "100%",
                            "height": "400px",
                            "display": "inline-block",
                            "border": "3px #5c5c5c solid",
                            "padding-top": "5px",
                            "padding-left": "1px",
                            "overflow": "hidden"
                        })
                     
                     
                     ])
                 
                 
                 
                 
                

                
                
                ])
        ])
       
    ])


@app.callback(
[Output('feature-graphic', 'figure'),
 Output('feature-graphic1', 'figure'), 
 Output('feature-graphic2', 'figure'), 
 Output('feature-graphic3', 'figure'), 
 Output('feature-graphic4', 'figure'), 
 Output('feature-graphic5', 'figure')],
 # Output('cm_multi', 'children'), 
 # Output('cm_overall', 'children')],

[Input('button', "n_clicks")],
[State('to_keep_classes', 'value'),
 State('frequency_target', 'value'),
 State('std_var_gene_seq_len', 'value'),
 State('number_unique_low_level', 'value'),
 State('skewness_coef', 'value'),
 State('method', 'value'), 
 State('ml_method', 'value'), 
 State('max_features', 'value'), 
 State('univariate_f1_score', 'value')])
def update_confusion_matrix(n_clicks,to_keep_classes, frequency_target, std_var_gene_seq_len, number_unique_low_level, skewness_coef, 
                            method, ml_method, max_features, univariate_f1_score):
    # data_classes = data[data["high_level_substr"].isin(to_keep_classes)]
    
    # if type(to_keep_classes) != type(None):
    if len(to_keep_classes) > 0:
        # data["binary"] = [1 if substr in to_keep_classes else 0 for substr in data["high_level_substr"]]
        selected_classes_high_level =  to_keep_classes
    
    # if type(to_keep_classes) == type(None):
    elif (type(frequency_target) != type(None)) & (type(std_var_gene_seq_len) != type(None)) & (type(number_unique_low_level) != type(None)) & (type(skewness_coef) != type(None)): 
        all_combined_df = all_combined[(all_combined["frequency"] >= frequency_target) & (all_combined["std_length"] <= std_var_gene_seq_len) & 
                                    (all_combined["number_unique"] <= number_unique_low_level) & 
                                    (all_combined["skewness"] >= skewness_coef) & (all_combined["5-fold-average-f1"] >= univariate_f1_score)]
        
        selected_classes_high_level = all_combined_df["high_level_substr"].value_counts().keys().tolist()
    else: 
        pass
        # data["binary"] = [1 if substr in classes_keep else 0 for substr in data["high_level_substr"]]
    
    # X_train, X_test, y_train, y_test = train_test_split(data["sig_gene_seq"], data[["binary", "high_level_substr"]], 
    #                                                     test_size=0.4, stratify = data["high_level_substr"])
    
    # index_for_one = (y_train["binary"] == 1)
    
    
    
    # X_train_multi = data["sig_gene_seq"]
    # y_train_multi = data["high_level_substr"]
    
    # index_for_one_test = (y_test["binary"] == 1)
    # X_test_multi = X_test[index_for_one_test]
    # y_test_multi = y_test[index_for_one_test]["high_level_substr"]
    
    if ml_method == "BRF": 
            
        rf = BalancedRandomForestClassifier(n_jobs = 6)
        rf1 = BalancedRandomForestClassifier(n_jobs = 6)
    else: 
        rf = RandomForestClassifier(n_jobs = 6)
        rf1 = RandomForestClassifier(n_jobs = 6)
    
    
    
    if method == "Method1":
        vectorizer_word = CountVectorizer(tokenizer=lambda x: str(x).replace("|", ",").split(','), lowercase = False, 
                                          max_features=max_features)
        selected_data_high_level = data[data["high_level_substr"].isin(selected_classes_high_level)]
        part1 = selected_data_high_level[["sig_gene_seq", "high_level_substr"]]
        other_part = data.iloc[~data.index.isin(part1.index.tolist())]
        other_part = other_part[["sig_gene_seq", "high_level_substr"]]
        other_part["high_level_substr"] = "others"
        combo = pd.concat([part1, other_part]).sample(frac = 1.0)
        skf = StratifiedKFold(n_splits=10)
        # brf = BalancedRandomForestClassifier()

        # pipeline
        clf = Pipeline([('vectorizer',vectorizer_word),
                ('brf',rf)])
        length = len(combo["high_level_substr"].value_counts())
        
        cm1 = np.zeros((length, length))
        unraveled_positions = []
        for train_index, test_index in skf.split(combo["sig_gene_seq"], combo["high_level_substr"]):

            X_train, X_test = combo[["sig_gene_seq"]].iloc[train_index,:], combo[["sig_gene_seq"]].iloc[test_index,:]
            y_train, y_test = combo[["high_level_substr"]].iloc[train_index,:], combo[["high_level_substr"]].iloc[test_index,:]
            clf.fit(X_train.values, y_train.values)
            y_pred_test = clf.predict(X_test.values)
            # get the array oaf confusion matrix
            cm = confusion_matrix(y_test, y_pred_test, normalize = 'true')
    
            unraveled_positions.append(cm.ravel().tolist())
    
            cm1 += cm  
    
        # get the array oaf confusion matrix
        cm = cm1/10

        # dataframe for confusion matrix
        df_cm = pd.DataFrame(cm, index = [i for i in clf.classes_],
                  columns = [i for i in clf.classes_])
        
        flattened_confusion_matrices = pd.DataFrame(unraveled_positions)
        df_cm_std = np.array(flattened_confusion_matrices.std(0)).reshape(df_cm.shape[1],df_cm.shape[1])
        df_cm_std = df_cm_std/np.sqrt(10)
        df_cm_std = pd.DataFrame(df_cm_std, index = [i for i in clf.classes_],
                  columns = [i for i in clf.classes_])
        
        
            
        return px.imshow(df_cm, text_auto=True, color_continuous_scale='RdBu_r', 
                         labels=dict(y="True Label", x="Predicted Label"), 
                         title = "Confusion Matrix for Single Stage Model (average accuracy)"), \
               px.imshow(df_cm_std, text_auto=True, color_continuous_scale='RdBu_r', 
                         labels=dict(y="True Label", x="Predicted Label"), 
                         title = "Confusion Matrix for Single Stage Model (standard errors)"), \
               px.imshow(df_cm, text_auto=True, color_continuous_scale='RdBu_r', 
                         labels=dict(y="True Label", x="Predicted Label"), 
                         title = "Confusion Matrix for Single Stage Model (average accuracy)"), \
               px.imshow(df_cm_std, text_auto=True, color_continuous_scale='RdBu_r', 
                         labels=dict(y="True Label", x="Predicted Label"), 
                         title = "Confusion Matrix for Single Stage Model (standard errors)"), \
               px.imshow(df_cm, text_auto=True, color_continuous_scale='RdBu_r', 
                         labels=dict(y="True Label", x="Predicted Label"), 
                         title = "Confusion Matrix for Single Stage Model (average accuracy)"), \
               px.imshow(df_cm_std, text_auto=True, color_continuous_scale='RdBu_r', 
                         labels=dict(y="True Label", x="Predicted Label"), 
                         title = "Confusion Matrix for Single Stage Model (standard errors)")
               
    if method == "Method2":
        data_all = data.copy()
        data_all["high_level_substr"] = [substr if substr in selected_classes_high_level else "others" for substr in data_all["high_level_substr"] ]
        cm1 = np.zeros((len(data_all["high_level_substr"].value_counts()), len(data_all["high_level_substr"].value_counts())))
        unraveled_positions = []
        
        unraveled_positions_bin = []
        cm1_binary = np.zeros((2,2))
        
        unraveled_positions_multi = []
        cm1_multi = np.zeros((len(selected_classes_high_level),len(selected_classes_high_level)))
        
        
        
        
        skf = StratifiedKFold(n_splits=10)
        for train_index, test_index in skf.split(data_all["sig_gene_seq"], data_all["high_level_substr"]):
            X_train, X_test = data_all[["sig_gene_seq"]].iloc[train_index,:], data_all[["sig_gene_seq"]].iloc[test_index,:]
            y_train, y_test = data_all[["high_level_substr"]].iloc[train_index,:], \
                              data_all[["high_level_substr"]].iloc[test_index,:]
    
            # first do the binary model
            y_train_binary = ["known" if item != "others" else "others" for item in y_train.values]
            y_test_binary = ["known" if item != "others" else "others" for item in y_test.values]
            # pipeline
            clf_binary = Pipeline([('vectorizer',CountVectorizer(tokenizer=lambda x: str(x).replace("|", ",").split(','), lowercase = False, 
                                                                 max_features=max_features)),
                                   ('brf',rf)])

            clf_binary.fit(X_train.values, y_train_binary)
            # multi class model
            combo_train = pd.concat([X_train, y_train],1)
    
            combo_train = combo_train[combo_train["high_level_substr"] != "others"]
    
            # get the material needed for multi class model
            X_train_multi, y_train_multi = combo_train["sig_gene_seq"], combo_train["high_level_substr"]
            
            combo_test1 = pd.concat([X_test, y_test],1)
    
            combo_test1 = combo_test1[combo_test1["high_level_substr"] != "others"]
    
            # get the material needed for multi class model
            X_test_multi, y_test_multi_red = combo_test1["sig_gene_seq"], combo_test1["high_level_substr"]
            
            
            
            clf_multi = Pipeline([('vectorizer',CountVectorizer(tokenizer=lambda x: str(x).replace("|", ",").split(','), lowercase = False, 
                                                                max_features=max_features)),
                ('brf',rf1)])
    
            clf_multi.fit(X_train_multi, y_train_multi)


            y_test_multi_red_pred = clf_multi.predict(X_test_multi.values)
            
            cm_multi = confusion_matrix(y_test_multi_red, y_test_multi_red_pred, normalize = 'true')
            
            unraveled_positions_multi.append(cm_multi.ravel().tolist())
            
            cm1_multi += cm_multi
    
    
    
            # get predictions for test and see
            y_test_pred_binary = clf_binary.predict(X_test.values)
            
            cm_binary = confusion_matrix(y_test_binary, y_test_pred_binary, normalize = 'true')
            
            unraveled_positions_bin.append(cm_binary.ravel().tolist())
            
            cm1_binary += cm_binary
            
            
            # get the predictions for the multi class
            y_test_multi = clf_multi.predict(X_test.values)
    
            # combined test
            combo_test = pd.DataFrame({"binary_pred": y_test_pred_binary, "multiclass_pred": y_test_multi, "actual_pred": y_test["high_level_substr"].values})
    
            combo_test["final_pred"] = ["others" if n == "others" else combo_test["multiclass_pred"][i] for i, n in enumerate(combo_test["binary_pred"])]
    
            cm = confusion_matrix(combo_test["actual_pred"], combo_test["final_pred"], normalize = 'true', 
                         labels = data_all["high_level_substr"].value_counts().keys().tolist())

            unraveled_positions.append(cm.ravel().tolist())
    
            cm1 += cm  
            
            # get the array oaf confusion matrix
        cm = cm1/10

        # dataframe for confusion matrix
        df_cm = pd.DataFrame(cm, index = [i for i in data_all["high_level_substr"].value_counts().keys().tolist()],
                  columns = [i for i in data_all["high_level_substr"].value_counts().keys().tolist()])
            
        flattened_confusion_matrices = pd.DataFrame(unraveled_positions)
        df_cm_std = np.array(flattened_confusion_matrices.std(0)).reshape(df_cm.shape[1],df_cm.shape[1])
        df_cm_std = df_cm_std/np.sqrt(10)
        df_cm_std = pd.DataFrame(df_cm_std, index = [i for i in data_all["high_level_substr"].value_counts().keys().tolist()],
                  columns = [i for i in data_all["high_level_substr"].value_counts().keys().tolist()])
        
        cm_bin = cm1_binary/10
        
        df_cm_bin = pd.DataFrame(cm_bin, index = [i for i in clf_binary.classes_],
                  columns = [i for i in clf_binary.classes_])
        
        
        flattened_confusion_matrices_bin = pd.DataFrame(unraveled_positions_bin)
        df_cm_std_bin = np.array(flattened_confusion_matrices_bin.std(0)).reshape(df_cm_bin.shape[1],df_cm_bin.shape[1])
        df_cm_std_bin = df_cm_std_bin/np.sqrt(10)
        df_cm_std_bin = pd.DataFrame(df_cm_std_bin, index = [i for i in clf_binary.classes_],
                  columns = [i for i in clf_binary.classes_])
        
        
        cm_multi = cm1_multi/10
        
        df_cm_multi = pd.DataFrame(cm_multi, index = [i for i in clf_multi.classes_],
                  columns = [i for i in clf_multi.classes_])
        
        
        flattened_confusion_matrices_multi = pd.DataFrame(unraveled_positions_multi)
        df_cm_std_multi = np.array(flattened_confusion_matrices_multi.std(0)).reshape(df_cm_multi.shape[1],
                                                                                      df_cm_multi.shape[1])
        df_cm_std_multi = df_cm_std_multi/np.sqrt(10)
        df_cm_std_multi = pd.DataFrame(df_cm_std_multi, index = [i for i in clf_multi.classes_],
                  columns = [i for i in clf_multi.classes_])

        return px.imshow(df_cm, text_auto=True, color_continuous_scale='RdBu_r', 
                         labels=dict(y="True Label", x="Predicted Label"), 
                         title = "Confusion Matrix for Two Stage Model (Average Accuracy)"), \
               px.imshow(df_cm_std, text_auto=True, color_continuous_scale='RdBu_r', 
                         labels=dict(y="True Label", x="Predicted Label"), 
                         title = "Confusion Matrix for Two Stage Model (Standard Errors)"), \
               px.imshow(df_cm_bin, text_auto=True, color_continuous_scale='RdBu_r', 
                         labels=dict(y="True Label", x="Predicted Label"), 
                         title = "Confusion Matrix for the Binary Classifier (Average Accuracy)"), \
               px.imshow(df_cm_std_bin, text_auto=True, color_continuous_scale='RdBu_r', 
                         labels=dict(y="True Label", x="Predicted Label"), 
                         title = "Confusion Matrix for the Binary Classifier (Standard Error)"), \
               px.imshow(df_cm_multi, text_auto=True, color_continuous_scale='RdBu_r', 
                         labels=dict(y="True Label", x="Predicted Label"), 
                         title = "Confusion Matrix for the Multiclass Classifier (Average Accuracy)"), \
               px.imshow(df_cm_std_multi, text_auto=True, color_continuous_scale='RdBu_r', 
                         labels=dict(y="True Label", x="Predicted Label"), 
                         title = "Confusion Matrix for the Multiclass Classifier (Standard Error)")
        
@app.callback(
[Output('gene_seq_similar', 'children'),
 Output('gene_similar', 'children')],
[Input('button1', "n_clicks")],
[State('gene_sequence', 'value'),
 State('gene', 'value')])
def retrieve_similar(n_clicks, gene_sequence, gene):
    table_gene_sequence = model1.dv.most_similar(model1.infer_vector(gene_sequence.split(",")))
    table_gene_sequence = pd.DataFrame(table_gene_sequence)
    indexes = table_gene_sequence[0]
    gene_seqs = updated_data_supervised.iloc[indexes,:].reset_index(drop = True)
    table_gene_sequence.iloc[:,0] = gene_seqs
    table_gene_sequence.columns = ["gene_sequence", "similarity"]
    table_gene = model1.wv.most_similar(gene)
    table_gene = pd.DataFrame(table_gene)
    table_gene.columns = ["gene", "similarity"]
    table_gene_sequence["similarity"], table_gene["similarity"] = round(table_gene_sequence["similarity"],3), round(table_gene["similarity"],3)
    return dbc.Table.from_dataframe(table_gene_sequence), dbc.Table.from_dataframe(table_gene)


if __name__ == '__main__':
    app.run_server(debug=True)