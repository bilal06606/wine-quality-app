import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

BASE = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(BASE / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(BASE / 'test.csv')


@st.cache_data
def test_proba(_model, features):
    return _model.predict_proba(features)[:, 1]


st.set_page_config(page_title='Оценка качества вина', page_icon='🍷', layout='wide')
model = load_model()
test = load_test()
features = test.drop(columns='target')
y_test = test['target']
feature_names = features.columns.tolist()
with (BASE / 'config.json').open(encoding='utf-8') as f:
    saved_threshold = float(json.load(f)['threshold'])

st.sidebar.header('Настройки')
threshold = st.sidebar.slider('Порог хорошего вина', min_value=0.0, max_value=1.0,
                              value=saved_threshold, step=0.001, key='threshold')
st.sidebar.caption('Порог единый для всех вкладок.')
st.title('🍷 Отбор хороших вин')
st.write('Модель оценивает вероятность того, что эксперт поставит вину не менее 7 баллов.')

single_tab, metrics_tab, batch_tab, coef_tab = st.tabs([
    'Одно вино', 'Модельные метрики', 'Пакетное предсказание', 'Коэффициенты'
])

with single_tab:
    st.subheader('Характеристики вина')
    entry = {}
    cols = st.columns(2)
    for i, name in enumerate(feature_names):
        if name == 'is_red':
            continue
        low = float(features[name].min())
        high = float(features[name].max())
        median = float(features[name].median())
        with cols[i % 2]:
            entry[name] = st.number_input(name, min_value=low, max_value=high,
                                          value=median, step=max((high - low) / 100, 0.00001),
                                          format='%.5f')
    entry['is_red'] = int(st.toggle('Красное вино (выкл. — белое)',
                                     value=bool(features['is_red'].median() > 0)))
    one_wine = pd.DataFrame([entry], columns=feature_names)
    one_prob = float(model.predict_proba(one_wine)[0, 1])
    st.metric('Вероятность хорошего вина', f'{one_prob:.1%}')
    st.success('Подходит в подборку' if one_prob >= threshold else 'Не подходит в подборку')

with metrics_tab:
    st.subheader('Качество на тестовой выборке')
    prob = test_proba(model, features)
    pred = (prob >= threshold).astype(int)
    precision = metrics.precision_score(y_test, pred, zero_division=0)
    recall = metrics.recall_score(y_test, pred, zero_division=0)
    f1 = metrics.f1_score(y_test, pred, zero_division=0)
    m1, m2, m3 = st.columns(3)
    m1.metric('Precision', f'{precision:.3f}')
    m2.metric('Recall', f'{recall:.3f}')
    m3.metric('F1', f'{f1:.3f}')

    fig_cm, ax_cm = plt.subplots(figsize=(4.8, 3.7))
    metrics.ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax_cm, cmap='Blues',
                                                     colorbar=False)
    ax_cm.set_title('Матрица ошибок')
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    fpr, tpr, _ = metrics.roc_curve(y_test, prob)
    pr, rc, _ = metrics.precision_recall_curve(y_test, prob)
    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    current_fpr = fp / (fp + tn) if fp + tn else 0
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr, label=f'ROC-AUC = {metrics.roc_auc_score(y_test, prob):.3f}')
    ax[0].scatter(current_fpr, recall, color='darkred', s=55, zorder=5, label='Текущий порог')
    ax[0].plot([0, 1], [0, 1], '--', color='gray')
    ax[0].set(xlabel='FPR', ylabel='TPR', title='ROC-кривая')
    ax[1].plot(rc, pr, label=f'PR-AUC = {metrics.average_precision_score(y_test, prob):.3f}')
    ax[1].scatter(recall, precision, color='darkred', s=55, zorder=5, label='Текущий порог')
    ax[1].axhline(y_test.mean(), linestyle='--', color='gray')
    ax[1].set(xlabel='Recall', ylabel='Precision', title='PR-кривая')
    for axis in ax:
        axis.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.info('При увеличении порога в подборку попадает меньше вин: обычно precision растёт, а recall падает. '
            'При уменьшении порога вин будет больше, но возрастёт риск рекомендовать посредственное.')

with batch_tab:
    st.subheader('Прогноз для CSV')
    file = st.file_uploader('Загрузите CSV с 12 признаками вина, без target', type='csv')
    if file is not None:
        try:
            uploaded = pd.read_csv(file)
            missing = [col for col in feature_names if col not in uploaded.columns]
            if missing:
                st.error('В файле не хватает столбцов: ' + ', '.join(missing))
            else:
                X_uploaded = uploaded.loc[:, feature_names].apply(pd.to_numeric, errors='raise')
                if X_uploaded.isna().any().any():
                    st.error('В признаках есть пустые значения. Заполните их перед загрузкой.')
                else:
                    p_upload = model.predict_proba(X_uploaded)[:, 1]
                    result = uploaded.copy()
                    result['probability_good'] = p_upload
                    result['prediction'] = (p_upload >= threshold).astype(int)
                    st.dataframe(result, use_container_width=True)
                    st.download_button('Скачать предсказания',
                                       data=result.to_csv(index=False).encode('utf-8-sig'),
                                       file_name='wine_predictions.csv', mime='text/csv')
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f'Не удалось прочитать CSV: {exc}')

with coef_tab:
    st.subheader('Влияние признаков')
    logistic = model.named_steps['logisticregression']
    coefs = pd.Series(logistic.coef_[0], index=feature_names).sort_values()
    fig_coef, ax_coef = plt.subplots(figsize=(8, 5))
    coefs.plot.barh(ax=ax_coef)
    ax_coef.axvline(0, color='gray', linewidth=1)
    ax_coef.set(xlabel='Коэффициент на стандартизованном признаке', ylabel='')
    st.pyplot(fig_coef)
    plt.close(fig_coef)
    st.write('За хорошее вино сильнее всего голосуют: **' + ', '.join(coefs.tail(3).index[::-1]) + '**.')
    st.write('Против хорошего вина: **' + ', '.join(coefs.head(3).index) + '**.')
    st.caption('Коэффициенты показывают вклад при остальных признаках без изменений, а не причинный эффект.')

