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

#model1 = gensim.models.doc2vec.Doc2Vec.load(r"D:\Gene_Project\Refined_Codes\doc2vec_on_unsupervised.model")

data = pd.read_csv(r"D://Gene_Project//pul_seq_low_high_substr_year_corrected.csv")
updated_data_supervised = pd.read_csv(r"D://Gene_Project//all_unsupervised_genes.csv")

dom_eigen_val = pd.read_csv(r"D://Gene_Project//dominant_eigen.csv")



# get the counts
# catch = []

# for substr in data["high_level_substr"].value_counts().keys().tolist():
#     data_substr = data[data["high_level_substr"] == substr]
#     purity = np.round(np.mean(data_substr["high_level_substr"] == data_substr["low_level_substr"]),2)
#     catch.append([substr, purity])
    
# catch_df = pd.DataFrame(catch)
# catch_df.columns = ["high_level_substr", "purity"]    

# catch_df.set_index("high_level_substr").reindex(data["high_level_substr"].value_counts().keys().tolist()).reset_index()
    
# get the frequency counts
D = data.high_level_substr.value_counts()
# convert to a dictionary
D = dict(D)

freq_data = pd.DataFrame({"high_level_substr":D.keys(), "frequency":D.values() })

fig = px.bar(x = D.keys(), y = list(D.values()), title = "Frequencies for the high level substrates")
# fig2 = px.bar(x = catch_df["high_level_substr"], y = catch_df["purity"])


data["length"] = [len(seq.replace("|",",").split(",")) for seq in data["sig_gene_seq"]]

D = dict(data.groupby("high_level_substr")["length"].std().reindex(D.keys()))

length_data = pd.DataFrame({"high_level_substr":D.keys(), "std_length":D.values() })



fig3 = px.bar(x = D.keys(), y = list(D.values()), title = "Standard Deviation for lengths of the gene sequences")

D = dict(data.groupby("high_level_substr")["low_level_substr"].nunique().reindex(D.keys()))

low_unique_data = pd.DataFrame({"high_level_substr":D.keys(), "num_unique_low_level_substr":D.values() })

all_combined = pd.merge(freq_data, length_data, how = "inner", on = "high_level_substr").merge(low_unique_data, how = "inner", on = "high_level_substr")
all_combined = pd.merge(all_combined, dom_eigen_val, how = "inner", on = "high_level_substr")


fig4 = px.bar(x = D.keys(), y = list(D.values()), title = "Number of unique low level substrates")


fig5 = px.bar(x = dom_eigen_val["high_level_substr"], y = dom_eigen_val["explained_var"],
              title = "Percentage variation in gene composition explained by the dominant eigenvector")



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


dropdown = []

for substr in data["high_level_substr"].value_counts().keys().tolist():
    dict1 = {"label": substr, "value": substr}
    dropdown.append(dict1)

method_dropdown = [{"label":"Method1", "value": "Method1"}, 
                   {"label":"Method2", "value": "Method2"}]

ml_dropdown = [{"label":"Random Forest", "value": "RF"}, 
                   {"label":"Balanced Random Forest", "value": "BRF"}]
   
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# unsup_seq = []
# for seq in updated_data_supervised["sequence"]: 
#     dict1 = {"label": seq, "value":seq}
#     unsup_seq.append(dict1)
    


# unsup_genes = []
# for seq in model1.wv.index_to_key: 
#     dict1 = {"label": seq, "value":seq}
#     unsup_genes.append(dict1)    
    

    
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
                            placeholder = "High Level Substrates"
                            # value = ["multiple_substrates", "xylan", "pectin"]
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
                        "width": "15%",
                        "margin-left": "5%",
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
                        "width": "15%",
                        "margin-left": "5%",
                      
                        "verticalAlign": "top"
                    }), 
                
                
                html.Div([
                    
                    dcc.Input(
                            id = "number_unique_low_level", 
                            type = "number", 
                            placeholder = "# unique less than",
                            value = ["multiple_substrates", "xylan", "pectin"]
                                )
                    
                    ],
                    style={
                        "display": "inline-block",
                        "width": "15%",
                        "margin-left": "5%",
                        "verticalAlign": "top"
                    }), 
                
                
                html.Div([
                    
                    dcc.Input(
                            id = "var_ratio_first_eig_vector", 
                            type = "number", 
                            placeholder = "Var ratio greater than"
                            # value = ["multiple_substrates", "xylan", "pectin"]
                                )
                    
                    ],
                    style={
                        "display": "inline-block",
                        "width": "15%",
                        "margin-left": "5%",
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
                        "margin-left": "5%",
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
                     
                     html.Div(id = "cm_multi", 
                               style={
                            "width": "100%",
                            "height": "300px",
                            "display": "inline-block",
                            "border": "3px #5c5c5c solid",
                            "padding-top": "5px",
                            "padding-left": "1px",
                            "overflow": "hidden"
                        })
                     
                     
                     ]), 
                 
                 html.Div([
                     
                     html.Div(id = "cm_overall", 
                               style={
                            "width": "100%",
                            "height": "300px",
                            "display": "inline-block",
                            "border": "3px #5c5c5c solid",
                            "padding-top": "5px",
                            "padding-left": "1px",
                            "overflow": "hidden"
                        })
                     
                     
                     ])
                 
                 
                 
                
                
                ])
     # dcc.Tab(label = "Search Engine for similar gene sequences and genes",
     #        children = [
                
     #             html.Div( ## Select menus
     #        [
     #            html.Div([
                    
     #                dcc.Input(
     #                        id = "gene_sequence", 
     #                        placeholder = "Enter gene sequence"
     #                        # value = ["multiple_substrates", "xylan", "pectin"]
     #                            )
                    
     #                ],
     #                style={
     #                    "display": "inline-block",
     #                    "width": "20%"
     #                }),
     #            html.Div([
                    
     #                dcc.Input(
     #                        id = "gene", 
     #                        placeholder = "Enter gene"
     #                            )
                    
     #                ],
     #                style={
     #                    "display": "inline-block",
     #                    "width": "30%",
     #                    "margin-left": "20px",
     #                    "verticalAlign": "top"
     #                }), 
                

                
     #            html.Div([
     #                html.Button('GO', id='button1')
     #                ])
                
     #        ]
     #    ), 
     #             html.Div([
                     
     #                 html.Div(id = "gene_seq_similar", 
     #                           style={
     #                        "width": "100%",
     #                        "height": "400px",
     #                        "display": "inline-block",
     #                        "border": "3px #5c5c5c solid",
     #                        "padding-top": "5px",
     #                        "padding-left": "1px",
     #                        "overflow": "hidden"
     #                    })
                     
                     
     #                 ]), 
                 
     #             html.Div([
                     
     #                 html.Div(id = "gene_similar", 
     #                           style={
     #                        "width": "100%",
     #                        "height": "400px",
     #                        "display": "inline-block",
     #                        "border": "3px #5c5c5c solid",
     #                        "padding-top": "5px",
     #                        "padding-left": "1px",
     #                        "overflow": "hidden"
     #                    })
                     
                     
     #                 ])
                 
                 
                 
                 
                

                
                
     #            ])
        ])
       
    ])


@app.callback(
[Output('feature-graphic', 'figure'),
 Output('feature-graphic1', 'figure'), 
 Output('cm_multi', 'children'), 
 Output('cm_overall', 'children')],

[Input('button', "n_clicks")],
[State('to_keep_classes', 'value'),
 State('frequency_target', 'value'),
 State('std_var_gene_seq_len', 'value'),
 State('number_unique_low_level', 'value'),
 State('var_ratio_first_eig_vector', 'value'),
 State('method', 'value'), 
 State('ml_method', 'value')])
def update_confusion_matrix(n_clicks,to_keep_classes, frequency_target, std_var_gene_seq_len, number_unique_low_level, var_ratio_first_eig_vector, 
                            method, ml_method):
    # data_classes = data[data["high_level_substr"].isin(to_keep_classes)]
    
    if type(to_keep_classes) != type(None):
        data["binary"] = [1 if substr in to_keep_classes else 0 for substr in data["high_level_substr"]]
    
    if type(to_keep_classes) == type(None): 
        all_combined_df = all_combined[(all_combined["frequency"] >= frequency_target) & (all_combined["std_length"] <= std_var_gene_seq_len) & 
                                    (all_combined["num_unique_low_level_substr"] <= number_unique_low_level) & 
                                    (all_combined["explained_var"] >= var_ratio_first_eig_vector)]
        
        classes_keep = all_combined_df["high_level_substr"].tolist()
        data["binary"] = [1 if substr in classes_keep else 0 for substr in data["high_level_substr"]]
    
    X_train, X_test, y_train, y_test = train_test_split(data["sig_gene_seq"], data[["binary", "high_level_substr"]], 
                                                        test_size=0.4, stratify = data["high_level_substr"])
    
    index_for_one = (y_train["binary"] == 1)
    X_train_multi = X_train[index_for_one]
    y_train_multi = y_train[index_for_one]["high_level_substr"]
    
    index_for_one_test = (y_test["binary"] == 1)
    X_test_multi = X_test[index_for_one_test]
    y_test_multi = y_test[index_for_one_test]["high_level_substr"]
    
    if ml_method == "BRF": 
            
        rf = BalancedRandomForestClassifier(n_jobs = 6)
    else: 
        rf = RandomForestClassifier(n_jobs = 6)
    
    
    vectorizer_word = CountVectorizer(tokenizer=lambda x: str(x).replace("|", ",").split(','), lowercase = False)
    if method == "Method1":
            
        # rf = BalancedRandomForestClassifier(n_jobs = 6)
        # pipeline
        clf = Pipeline([('vectorizer',vectorizer_word),
                            ('rf',rf)])
        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        param_grid = {
                        'vectorizer__min_df': [1,2],
                        'rf__n_estimators': [100,200,400], 
                        'rf__max_features': ["auto", "log2"]
                    }
        # fit the search
        search_binary = GridSearchCV(clf, param_grid, n_jobs=6 , verbose = 3, cv = 5, scoring = "balanced_accuracy")
        search_binary.fit(X_train, y_train["binary"])
        y_test_pred = search_binary.predict(X_test)
        # get the array oaf confusion matrix
        cm = confusion_matrix(y_test["binary"], y_test_pred, normalize = 'true')

        # dataframe for confusion matrix
        df_cm_binary = pd.DataFrame(cm, index = [i for i in search_binary.classes_],
                 columns = [i for i in search_binary.classes_])
            
        # rf = BalancedRandomForestClassifier(n_jobs = 6)
        # pipeline
        clf = Pipeline([('vectorizer',vectorizer_word),
                            ('rf',rf)])
        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        param_grid = {
                        'vectorizer__min_df': [1,2],
                        'rf__n_estimators': [100,200,400], 
                        'rf__max_features': ["auto", "log2"]
                    }
        # fit the search
        search_multi = GridSearchCV(clf, param_grid, n_jobs=6 , verbose = 3, cv = 5, scoring = "balanced_accuracy")
        search_multi.fit(X_train_multi, y_train_multi)
        y_test_pred = search_multi.predict(X_test_multi)
        # get the array oaf confusion matrix
        cm = confusion_matrix(y_test_multi, y_test_pred, normalize = 'true')

        # dataframe for confusion matrix
        df_cm_multi = pd.DataFrame(cm, index = [i for i in search_multi.classes_],
                 columns = [i for i in search_multi.classes_])
            
        
        cm_multi =pd.DataFrame(classification_report(y_test_multi, y_test_pred, output_dict = True)).T.round(3).reset_index()
        cm_multi.columns = ["high_level_substr", 'precision', 'recall', 'f1-score', 'support']
        classwise, overall =cm_multi.iloc[:-3,:], cm_multi.iloc[-3:,:]
        overall.columns = ["average-type", 'precision', 'recall', 'f1-score', 'support']
            
        return px.imshow(df_cm_binary, text_auto=True, color_continuous_scale='RdBu_r'), \
               px.imshow(df_cm_multi, text_auto=True, color_continuous_scale='RdBu_r'), \
                  dbc.Table.from_dataframe(classwise), dbc.Table.from_dataframe(overall)
                  
            # plt.show()


# @app.callback(
# [Output('gene_seq_similar', 'children'),
#  Output('gene_similar', 'children')],
# [Input('button1', "n_clicks")],
# [State('gene_sequence', 'value'),
#  State('gene', 'value')])
# def retrieve_similar(n_clicks, gene_sequence, gene):
#     table_gene_sequence = model1.dv.most_similar(model1.infer_vector(gene_sequence.split(",")))
#     table_gene_sequence = pd.DataFrame(table_gene_sequence)
#     indexes = table_gene_sequence[0]
#     gene_seqs = updated_data_supervised.iloc[indexes,:].reset_index(drop = True)
#     table_gene_sequence.iloc[:,0] = gene_seqs
#     table_gene_sequence.columns = ["gene_sequence", "similarity"]
#     table_gene = model1.wv.most_similar(gene)
#     table_gene = pd.DataFrame(table_gene)
#     table_gene.columns = ["gene", "similarity"]
#     table_gene_sequence["similarity"], table_gene["similarity"] = round(table_gene_sequence["similarity"],3), round(table_gene["similarity"],3)
#     return dbc.Table.from_dataframe(table_gene_sequence), dbc.Table.from_dataframe(table_gene)


if __name__ == '__main__':
    app.run_server(debug=True)