
# ================================================================
# NexaCloud Churn Prediction API — app.py
# Run: uvicorn app:app --reload
# Docs: http://localhost:8000/docs
# ================================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional
import joblib
import numpy as np
import pandas as pd
import os

# ── Load artefacts ──────────────────────────────────────────────
MODEL     = joblib.load('model_artefacts/xgb_churn_model.pkl')
SCALER    = joblib.load('model_artefacts/scaler.pkl')
FEATURES  = joblib.load('model_artefacts/selected_features.pkl')
THRESHOLD = joblib.load('model_artefacts/threshold.pkl')

# ── App setup ───────────────────────────────────────────────────
app = FastAPI(
    title='NexaCloud Churn Prediction API',
    description='Predict whether a NexaCloud SaaS account is at risk of churning.',
    version='1.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

# ── Request schema ──────────────────────────────────────────────
class AccountFeatures(BaseModel):
    subscription_months:        int   = Field(..., ge=0,    description='Months account has been active')
    mrr_usd:                   float = Field(..., ge=0,    description='Monthly Recurring Revenue (USD)')
    ltv_usd:                   float = Field(..., ge=0,    description='Lifetime Value accumulated (USD)')
    num_addons:                int   = Field(..., ge=0, le=7, description='Number of add-ons enabled (0-7)')
    billing_cycle:             Literal['Monthly', 'Annual', 'Biennial']
    payment_method:            Literal['Electronic check', 'Mailed check',
                                        'Bank transfer (automatic)', 'Credit card (automatic)']
    plan_type:                 Literal['Free', 'Starter', 'Pro']
    enterprise_tier:           Literal['SMB', 'Enterprise']
    sso_enabled:               Literal['Yes', 'No']
    auto_backup_enabled:       Literal['Yes', 'No']
    endpoint_security_enabled: Literal['Yes', 'No']
    priority_support_enabled:  Literal['Yes', 'No']
    live_collab_enabled:       Literal['Yes', 'No']
    media_vault_enabled:       Literal['Yes', 'No']
    has_crm_integration:       Literal['Yes', 'No']
    e_invoicing_enabled:       Literal['Yes', 'No']

# ── Response schema ─────────────────────────────────────────────
class ChurnPrediction(BaseModel):
    account_churn_risk:   str
    churn_probability:    float
    threshold_used:       float
    risk_level:          str
    recommendation:      str

# ── Feature builder ─────────────────────────────────────────────
def build_features(data: AccountFeatures) -> pd.DataFrame:
    d = data.dict()

    # Maps
    yn = lambda v: 1 if v == 'Yes' else 0
    billing_risk_map = {'Biennial': 0, 'Annual': 1, 'Monthly': 2}
    payment_risk_map = {
        'Electronic check': 3, 'Mailed check': 2,
        'Bank transfer (automatic)': 1, 'Credit card (automatic)': 1
    }
    plan_risk_map = {'Free': 0, 'Starter': 1, 'Pro': 2}

    sm = d['subscription_months']
    mrr = d['mrr_usd']
    ltv = d['ltv_usd']
    n_add = d['num_addons']

    avg_monthly = ltv / sm if sm > 0 else mrr
    billing_risk = billing_risk_map[d['billing_cycle']]
    pay_risk = payment_risk_map[d['payment_method']]
    is_monthly = 1 if d['billing_cycle'] == 'Monthly' else 0
    is_elec = 1 if d['payment_method'] == 'Electronic check' else 0
    is_pro = 1 if d['plan_type'] == 'Pro' else 0
    is_first_year = 1 if sm <= 12 else 0
    is_long_term = 1 if sm >= 24 else 0
    has_sec_bundle = 1 if (yn(d['sso_enabled']) and yn(d['endpoint_security_enabled'])) else 0
    has_col_bundle = 1 if (yn(d['live_collab_enabled']) and yn(d['media_vault_enabled'])) else 0

    row = {
        'subscription_months'     : sm,
        'mrr_usd'                : mrr,
        'ltv_usd'                : ltv,
        'num_addons'             : n_add,
        'is_first_year'          : is_first_year,
        'is_very_new'            : 1 if sm <= 3 else 0,
        'is_long_term'           : is_long_term,
        'is_contract_end'        : 1 if sm % 12 == 0 else 0,
        'avg_monthly_revenue'    : avg_monthly,
        'mrr_to_ltv_ratio'       : mrr / (ltv + 1),
        'charge_increase_flag'   : 1 if mrr > avg_monthly else 0,
        'ltv_projection'         : ltv + mrr * 6,
        'price_to_tenure_ratio'  : mrr / (sm + 1),
        'billing_efficiency'     : ltv / (mrr * sm + 1),
        'mrr_tier'               : min(int(mrr // 35), 3),
        'addon_adoption_rate'    : n_add / (sm + 1),
        'has_security_bundle'    : has_sec_bundle,
        'has_collab_bundle'      : has_col_bundle,
        'has_backup_support'     : 1 if (yn(d['auto_backup_enabled']) and yn(d['priority_support_enabled'])) else 0,
        'is_pro_plan'            : is_pro,
        'is_free_plan'           : 1 if d['plan_type'] == 'Free' else 0,
        'is_starter_plan'        : 1 if d['plan_type'] == 'Starter' else 0,
        'has_crm_or_sub'         : 1 if (yn(d['has_crm_integration']) or 1) else 0,
        'billing_risk'           : billing_risk,
        'is_monthly_billing'     : is_monthly,
        'payment_risk_score'     : pay_risk,
        'is_electronic_check'    : is_elec,
        'paperless_electronic_risk': 1 if (yn(d['e_invoicing_enabled']) and is_elec) else 0,
        'is_enterprise'          : 1 if d['enterprise_tier'] == 'Enterprise' else 0,
        'high_cost_new_account'  : 1 if (mrr > 55 and sm < 12) else 0,
        'pro_monthly_risk'       : 1 if (is_pro and is_monthly) else 0,
        'new_account_elec_check' : 1 if (sm < 6 and is_elec) else 0,
        'monthly_no_addons'      : 1 if (is_monthly and n_add == 0) else 0,
        'longterm_no_security'   : 1 if (is_long_term and not has_sec_bundle) else 0,
        'many_addons_very_new'   : 1 if (n_add > 3 and sm <= 3) else 0,
        'engagement_score'       : n_add*0.35 + (2-billing_risk)*0.40 + min(sm/72, 1)*0.25,
        'churn_risk_score'       : is_monthly*0.30 + is_elec*0.20 + is_first_year*0.20
                                   + (1 if mrr>70 else 0)*0.15 + (1 if n_add==0 else 0)*0.15,
    }

    df_row = pd.DataFrame([row])
    # Align to selected features
    for col in FEATURES:
        if col not in df_row.columns:
            df_row[col] = 0
    return df_row[FEATURES]

# ── Endpoints ───────────────────────────────────────────────────
@app.get('/')
def root():
    return {
        'message': 'NexaCloud Churn Prediction API',
        'version': '1.0.0',
        'docs': '/docs',
        'health': '/health'
    }

@app.get('/health')
def health():
    return {'status': 'healthy', 'model': 'XGBoost', 'threshold': THRESHOLD}

@app.post('/predict', response_model=ChurnPrediction)
def predict_churn(account: AccountFeatures):
    try:
        features = build_features(account)
        features_scaled = SCALER.transform(features)
        prob = float(MODEL.predict_proba(features_scaled)[0, 1])
        churn = prob >= THRESHOLD

        if prob >= 0.75:
            risk_level = 'CRITICAL'
            recommendation = 'Immediate CSM intervention required. Offer retention package.'
        elif prob >= 0.5:
            risk_level = 'HIGH'
            recommendation = 'Schedule proactive check-in. Offer add-on trial or billing upgrade.'
        elif prob >= THRESHOLD:
            risk_level = 'MEDIUM'
            recommendation = 'Monitor closely. Send engagement nudge email within 7 days.'
        else:
            risk_level = 'LOW'
            recommendation = 'Account is healthy. Continue standard customer success motion.'

        return ChurnPrediction(
            account_churn_risk = 'CHURN RISK' if churn else 'RETAINED',
            churn_probability  = round(prob, 4),
            threshold_used     = round(THRESHOLD, 4),
            risk_level         = risk_level,
            recommendation     = recommendation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/predict/batch')
def predict_batch(accounts: list[AccountFeatures]):
    return [predict_churn(a) for a in accounts]
