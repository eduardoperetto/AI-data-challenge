import os
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from rulefit import RuleFit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, train_test_split

USING_RESULT_AS_DIFF_FROM_LAST = True
CONSIDERING_RTT_TR = False

# --- Caminhos e nomes de arquivos atualizados para RuleFit ---
BEST_PARAMS_RULEFIT_PATH = "best_params_rulefit_gb.joblib" if CONSIDERING_RTT_TR else "best_params_rulefit_gb_wortt.joblib"

def load_best_params_rulefit():
    """Carrega os melhores parâmetros para o RuleFit de um arquivo joblib."""
    if os.path.exists(BEST_PARAMS_RULEFIT_PATH):
        return joblib.load(BEST_PARAMS_RULEFIT_PATH)
    return {}

def save_best_params_rulefit(d):
    """Salva os melhores parâmetros para o RuleFit em um arquivo joblib."""
    joblib.dump(d, BEST_PARAMS_RULEFIT_PATH)

def load_data(input_csv):
    """Carrega os dados do CSV de entrada."""
    df = pd.read_csv(input_csv)
    X = df.iloc[:, :-4].values
    y = df.iloc[:, -4:].values
    return X, y, df.columns[:-4]

def split_data(X, y, seed):
    """Divide os dados em conjuntos de treino e teste."""
    return train_test_split(X, y, test_size=0.25, random_state=seed)

def create_main_output_folder():
    """Cria a pasta principal de saída para os resultados do RuleFit."""
    base_or_enriched = 'enriched' if CONSIDERING_RTT_TR else 'base'
    folder = os.path.join("output", "saving_csv", base_or_enriched, "rulefit_gb", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(folder, exist_ok=True)
    return folder

def get_best_params_for_column_rulefit(X_train, y_train, feature_names, seed, column_id, best_params_dict):
    """Executa GridSearchCV para encontrar os melhores hiperparâmetros para o RuleFit."""
    if column_id in best_params_dict:
        print(f"Usando parâmetros de cache para a coluna {column_id}")
        return best_params_dict[column_id]

    gbr = GradientBoostingRegressor(random_state=seed)
    rf_model = RuleFit(tree_generator=gbr, random_state=seed)
    
    param_grid = {
        "Cs": [100, 500], 
        "max_rules": [100, 500],
        "tree_generator__n_estimators": [100, 500],
        "tree_generator__max_depth": [3, 5],
        "tree_generator__learning_rate": [0.01, 0.1],
        "tree_generator__subsample": [0.75, 1.0]
    }
    
    start_time = time.perf_counter()
    fit_params = {'feature_names': feature_names}
    gs = GridSearchCV(
        rf_model,
        param_grid,
        cv=2,
        scoring="neg_mean_absolute_error",
        n_jobs=2,
        verbose=1
    )
    gs.fit(X_train, y_train, **fit_params)
    
    elapsed_time = time.perf_counter() - start_time
    print(f"Tempo de busca de hiperparâmetros para a coluna {column_id}: {elapsed_time:.3f}s")
    
    best_params_dict[column_id] = gs.best_params_
    save_best_params_rulefit(best_params_dict)
    return best_params_dict[column_id]

def build_rulefit_with_params(best_params, seed):
    """Constrói um modelo RuleFit com os melhores parâmetros encontrados."""
    gbr_params = {
        k.split('__')[1]: v for k, v in best_params.items() 
        if k.startswith('tree_generator__')
    }
    rulefit_params = {
        k: v for k, v in best_params.items() 
        if not k.startswith('tree_generator__')
    }

    gbr = GradientBoostingRegressor(random_state=seed, **gbr_params)
    model = RuleFit(tree_generator=gbr, random_state=seed, **rulefit_params)
    return model

def evaluate_model(model, X_test, y_test, idx_last_rate, idx_last_rate_std, csv_name):
    """Avalia o modelo, com a mesma lógica de ajuste de predição de antes."""
    y_pred = model.predict(X_test)
    if USING_RESULT_AS_DIFF_FROM_LAST:
        y_pred_adj = []
        y_test_adj = []
        for x_row, real_val, pred_val in zip(X_test, y_test, y_pred):
            last_mean = x_row[idx_last_rate]
            last_std = x_row[idx_last_rate_std]
            diff_base = last_std if ("std" in csv_name) else last_mean
            y_pred_adj.append(pred_val + diff_base)
            y_test_adj.append(real_val + diff_base)
        y_pred = y_pred_adj
        y_test = y_test_adj
    return mean_absolute_percentage_error(y_test, y_pred)

def rule_importance_string(model):
    """Extrai e formata as regras mais importantes (top 20) para o CSV."""
    rules = model.get_rules()
    rules = rules[rules.coef != 0].sort_values("importance", ascending=False)
    
    rules = rules.head(20)
    if rules.empty:
        return "NO_RULES_FOUND"

    rules['rule_cleaned'] = rules['rule'].str.replace('\n', ' ').str.replace(';', ',')
    
    return "<>".join([
        f"{row.rule_cleaned} (type={row.type}, coef={row.coef:.4f}, sup={row.support:.2f}, imp={row.importance:.4f})"
        for _, row in rules.iterrows()
    ])

def save_full_rule_importance_to_txt(model, target_name, folder_path):
    """
    Extrai todas as regras com coeficientes não nulos do modelo RuleFit
    e as salva em um arquivo .txt formatado.
    """
    # 1. Obter as regras
    rules_df = model.get_rules()

    # 2. Filtrar por regras e features com coeficiente não nulo e ordenar por importância
    important_rules_df = rules_df[rules_df['coef'] != 0].sort_values(by="importance", ascending=False)

    # 3. Construir o caminho do arquivo
    file_name = f"{target_name}_rule_importance.txt"
    file_path = os.path.join(folder_path, file_name)

    # 4. Salvar o DataFrame em um arquivo de texto
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Relatório Completo de Regras e Importância para o Alvo: {target_name}\n")
        f.write("="*80 + "\n")
        # Usar to_string() para uma saída bem formatada, sem truncar
        f.write(important_rules_df.to_string())

    print(f"Relatório completo de regras salvo em: {file_path}")


def save_csv(csv_path, line):
    """Salva uma linha de resultados no arquivo CSV, criando o cabeçalho se necessário."""
    exists = os.path.exists(csv_path)
    header = "Model;MeanOrStd?;ConsiderRTT_TR?;MAPE;Seed;RuleImportance;MaxRules;Cs;NumEstimators;MaxDepth;LearningRate;Subsample;TrainTime\n"
    with open(csv_path, "a" if exists else "w", encoding="utf-8") as f:
        if not exists:
            f.write(header)
        f.write(line + "\n")

def execute_rulefit_training_by_column(X_train, X_test, y_train, y_test, feature_names, column, csv_name, seed, folder, best_params_dict, idx_last_rate, idx_last_rate_std):
    """Orquestra o treino e avaliação do RuleFit para uma única coluna-alvo."""
    col_best_params = get_best_params_for_column_rulefit(X_train, y_train, feature_names, seed, column, best_params_dict)
    
    model = build_rulefit_with_params(col_best_params, seed)
    start_fit = time.perf_counter()
    model.fit(X_train, y_train, feature_names=feature_names)
    fit_time = time.perf_counter() - start_fit
    
    # --- CHAMADA DA NOVA FUNÇÃO ---
    # Salva o relatório completo de regras em um arquivo .txt separado
    column_name = csv_name.replace('.csv', '')
    save_full_rule_importance_to_txt(model, column_name, folder)
    
    mape = evaluate_model(model, X_test, y_test, idx_last_rate, idx_last_rate_std, csv_name)
    feats = rule_importance_string(model)
    
    line = ";".join([
        "RuleFit_GB",
        csv_name.replace(".csv", ""),
        str(CONSIDERING_RTT_TR),
        f"{mape:.6f}",
        str(seed),
        feats,
        str(col_best_params.get("max_rules", "")),
        str(col_best_params.get("Cs", "")),
        str(col_best_params.get("tree_generator__n_estimators", "")),
        str(col_best_params.get("tree_generator__max_depth", "")),
        str(col_best_params.get("tree_generator__learning_rate", "")),
        str(col_best_params.get("tree_generator__subsample", "")),
        f"{fit_time:.4f}"
    ])
    
    save_csv(os.path.join(folder, csv_name), line)
    
    save_model(model, column_name, seed, mape, folder)

def save_model(model, column, seed, mape, folder):
    """Salva o modelo treinado em um arquivo .pkl."""
    mape_str = f"{mape * 100:.2f}%"
    model_file = f"RuleFit_{column}_{seed}_{mape_str}.pkl"
    model_path = os.path.join(folder, model_file)
    joblib.dump({'model': model}, model_path)
    print(f"Modelo salvo em {model_path}")

def run_once(seed_folder, seed):
    """Executa um ciclo completo de treino e avaliação para uma única seed."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd() 
        
    base_csv = os.path.join(script_dir, "../prepared_data.csv")
    X, y, feature_names = load_data(base_csv)
    
    feature_names_list = list(feature_names)
    idx_last_rate = feature_names_list.index('dash_last_rate')
    idx_last_rate_std = feature_names_list.index('dash_last_rate_std')
    
    X_train, X_test, y_train_all, y_test_all = train_test_split(X, y, test_size=0.25, random_state=seed)
    
    pd.DataFrame(X_train, columns=feature_names).to_csv(os.path.join(seed_folder, "X_train.csv"), index=False)
    pd.DataFrame(X_test, columns=feature_names).to_csv(os.path.join(seed_folder, "X_test.csv"), index=False)
    
    targets = {-4: "mean_1", -3: "std_1", -2: "mean_2", -1: "std_2"}
    
    for col, name in targets.items():
        y_train_col = y_train_all[:, col]
        y_test_col = y_test_all[:, col]
        pd.DataFrame(y_train_col, columns=[name]).to_csv(os.path.join(seed_folder, f"y_train_{name}.csv"), index=False)
        pd.DataFrame(y_test_col, columns=[name]).to_csv(os.path.join(seed_folder, f"y_test_{name}.csv"), index=False)
    
    best_params_dict = load_best_params_rulefit()
    
    for col, name in targets.items():
        y_train_col = y_train_all[:, col]
        y_test_col = y_test_all[:, col]
        
        execute_rulefit_training_by_column(
            X_train, X_test, y_train_col, y_test_col,
            feature_names, col, f"{name}.csv", seed, seed_folder,
            best_params_dict, idx_last_rate, idx_last_rate_std
        )

def loop_exec(n):
    """Função principal que gerencia o loop de execuções com diferentes seeds."""
    main_folder = create_main_output_folder()
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    prepare_py_path = os.path.join(script_dir, "../prepare.py")
    if os.path.exists(prepare_py_path):
        with open(prepare_py_path, 'r') as f:
            lines = f.readlines()
        with open(os.path.join(main_folder, "params.txt"), 'w') as f_out:
            f_out.writelines(lines[7:50])
    
    seeds = [665, 616, 617, 232, 84, 230, 383, 887, 617, 531, 496]
    for seed in seeds:
        print("-" * 50)
        print(f"Executando para a Seed: {seed}")
        print("-" * 50)
        seed_folder = os.path.join(main_folder, str(seed))
        os.makedirs(seed_folder, exist_ok=True)
        # Passando a pasta da seed para a função run_once
        run_once(seed_folder, seed)
    
    print("\n" + "=" * 50)
    print("Resultados Finais (MAPE Médio ± Desvio Padrão)")
    print("=" * 50)
    for target in ["mean_1", "std_1", "mean_2", "std_2"]:
        mape_list = []
        for seed in seeds:
            seed_folder = os.path.join(main_folder, str(seed))
            target_file = os.path.join(seed_folder, f"{target}.csv")
            if not os.path.exists(target_file):
                continue
            df = pd.read_csv(target_file, sep=';')
            mape_list.append(df['MAPE'].iloc[0])
        
        if mape_list:
            mean_mape = np.mean(mape_list)
            std_mape = np.std(mape_list, ddof=1)
            print(f"{target}: {mean_mape:.6f} ± {std_mape:.6f}")

if __name__ == "__main__":
    loop_exec(1)