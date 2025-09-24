import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# Configure page
st.set_page_config(
    page_title="FinScope Global - Financial Inclusion Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced custom CSS
st.markdown("""
<style>
    :root {
        --bg-color: #ffffff;
        --text-color: #000000;
        --card-bg: #ffffff;
        --border-color: #e0e0e0;
    }
    
    [data-theme="dark"] {
        --bg-color: #0e1117;
        --text-color: #ffffff;
        --card-bg: #262730;
        --border-color: #404040;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    }
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border-left: 6px solid #667eea;
        margin-bottom: 20px;
        transition: all 0.3s ease;
        text-align: center;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }
    .demographic-showcase {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 2px solid #e9ecef;
    }
    .demo-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .demo-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--accent-color);
    }
    .demo-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .champion-card { --accent-color: #27AE60; box-shadow: 0 4px 15px rgba(39,174,96,0.1); }
    .priority-card { --accent-color: #E74C3C; box-shadow: 0 4px 15px rgba(231,76,60,0.1); }
    .recommendation-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border-left: 6px solid;
    }
    .action-timeline {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    .action-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .action-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    .immediate { border-left-color: #E74C3C; }
    .medium-term { border-left-color: #F39C12; }
    .long-term { border-left-color: #27AE60; }
    .region-info-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        border-left: 6px solid;
        animation: slideIn 0.5s ease-out;
    }
    .kpi-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 6px 20px rgba(102,126,234,0.3);
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    .stButton > button {
        width: 100%;
        border-radius: 25px;
        height: 3.5em;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102,126,234,0.4);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    # Regional data
    regional_data = {
        'region': [
            'High income',
            'East Asia & Pacific (excluding high income)',
            'Europe & Central Asia (excluding high income)',
            'South Asia (excluding high income)',
            'Latin America & Caribbean (excluding high income)',
            'Sub-Saharan Africa (excluding high income)',
            'Middle East & North Africa (excluding high income)'
        ],
        'inclusion_rate': [0.858, 0.568, 0.554, 0.483, 0.480, 0.427, 0.382],
        'count': [2938, 521, 1139, 352, 970, 1833, 558],
        'std': [0.173, 0.272, 0.221, 0.253, 0.202, 0.224, 0.230]
    }
    
    # Income group data
    income_data = {
        'income_group': ['High income', 'Upper middle income', 'Lower middle income', 'Low income'],
        'inclusion_rate': [0.870, 0.571, 0.440, 0.374],
        'count': [2790, 2203, 2328, 990]
    }
    
    # Random Forest feature importance
    feature_importance = {
        'feature': [
            'Business Loan Source', 'Business Loan Access', 'Emergency Funds', 'Digital Engagement Score',
            'Government Services Score', 'Loan Purpose Group', 'Mobile Payment S/R',
            'Prefer Digital Finance', 'Financial Activity Score', 'Income Digital Interaction',
            'Saved Any', 'Borrowed Any', 'Saved for Purchase', 'Prefer Digital Account'
        ],
        'importance': [0.1683, 0.1230, 0.0980, 0.0636, 0.0597, 0.0409, 0.0404, 
                      0.0392, 0.0390, 0.0378, 0.0351, 0.0273, 0.0251, 0.0250]
    }
    
    # Comprehensive demographic data across all regions
    regional_demographic_data = {
        'In Labor Force': {
            'High income': 0.930,
            'Europe & Central Asia (excluding high income)': 0.704,
            'East Asia & Pacific (excluding high income)': 0.597,
            'South Asia (excluding high income)': 0.560,
            'Latin America & Caribbean (excluding high income)': 0.569,
            'Middle East & North Africa (excluding high income)': 0.540,
            'Sub-Saharan Africa (excluding high income)': 0.500
        },
        'Men': {
            'High income': 0.888,
            'Europe & Central Asia (excluding high income)': 0.596,
            'East Asia & Pacific (excluding high income)': 0.576,
            'South Asia (excluding high income)': 0.533,
            'Latin America & Caribbean (excluding high income)': 0.527,
            'Middle East & North Africa (excluding high income)': 0.476,
            'Sub-Saharan Africa (excluding high income)': 0.468
        },
        'Women': {
            'High income': 0.855,
            'Europe & Central Asia (excluding high income)': 0.540,
            'East Asia & Pacific (excluding high income)': 0.576,
            'South Asia (excluding high income)': 0.435,
            'Latin America & Caribbean (excluding high income)': 0.454,
            'Middle East & North Africa (excluding high income)': 0.300,
            'Sub-Saharan Africa (excluding high income)': 0.383
        },
        'Rich 60%': {
            'High income': 0.903,
            'Europe & Central Asia (excluding high income)': 0.626,
            'East Asia & Pacific (excluding high income)': 0.637,
            'South Asia (excluding high income)': 0.537,
            'Latin America & Caribbean (excluding high income)': 0.559,
            'Middle East & North Africa (excluding high income)': 0.448,
            'Sub-Saharan Africa (excluding high income)': 0.497
        },
        'Poor 40%': {
            'High income': 0.834,
            'Europe & Central Asia (excluding high income)': 0.484,
            'East Asia & Pacific (excluding high income)': 0.484,
            'South Asia (excluding high income)': 0.401,
            'Latin America & Caribbean (excluding high income)': 0.381,
            'Middle East & North Africa (excluding high income)': 0.304,
            'Sub-Saharan Africa (excluding high income)': 0.333
        },
        'Urban': {
            'High income': 0.891,
            'Europe & Central Asia (excluding high income)': 0.769,
            'East Asia & Pacific (excluding high income)': 0.766,
            'South Asia (excluding high income)': 0.607,
            'Latin America & Caribbean (excluding high income)': 0.642,
            'Middle East & North Africa (excluding high income)': 0.491,
            'Sub-Saharan Africa (excluding high income)': 0.645
        },
        'Rural': {
            'High income': 0.879,
            'Europe & Central Asia (excluding high income)': 0.707,
            'East Asia & Pacific (excluding high income)': 0.670,
            'South Asia (excluding high income)': 0.596,
            'Latin America & Caribbean (excluding high income)': 0.556,
            'Middle East & North Africa (excluding high income)': 0.392,
            'Sub-Saharan Africa (excluding high income)': 0.516
        },
        'Higher Education': {
            'High income': 0.897,
            'Europe & Central Asia (excluding high income)': 0.638,
            'East Asia & Pacific (excluding high income)': 0.677,
            'South Asia (excluding high income)': 0.597,
            'Latin America & Caribbean (excluding high income)': 0.564,
            'Middle East & North Africa (excluding high income)': 0.425,
            'Sub-Saharan Africa (excluding high income)': 0.570
        },
        'Primary Education': {
            'High income': 0.769,
            'Europe & Central Asia (excluding high income)': 0.390,
            'East Asia & Pacific (excluding high income)': 0.457,
            'South Asia (excluding high income)': 0.419,
            'Latin America & Caribbean (excluding high income)': 0.368,
            'Middle East & North Africa (excluding high income)': 0.330,
            'Sub-Saharan Africa (excluding high income)': 0.337
        },
        'Age 15-24': {
            'High income': 0.781,
            'Europe & Central Asia (excluding high income)': 0.428,
            'East Asia & Pacific (excluding high income)': 0.543,
            'South Asia (excluding high income)': 0.431,
            'Latin America & Caribbean (excluding high income)': 0.403,
            'Middle East & North Africa (excluding high income)': 0.267,
            'Sub-Saharan Africa (excluding high income)': 0.361
        }
    }

    # Country mapping with ISO codes for proper map visualization
    country_mapping = {
        'High income': {
            'countries': [
                'Australia','Austria','Bahrain','Belgium','Canada','Chile','Croatia','Cyprus','Czechia',
                'Denmark','Estonia','Finland','France','Germany','Greece','Hong Kong SAR, China','Hungary',
                'Iceland','Ireland','Israel','Italy','Japan','Korea, Rep.','Kuwait','Latvia','Lithuania',
                'Luxembourg','Malta','Netherlands','New Zealand','Norway','Oman','Panama','Poland',
                'Portugal','Puerto Rico','Qatar','Romania','Saudi Arabia','Singapore','Slovak Republic',
                'Slovenia','Spain','Sweden','Switzerland','Taiwan, China','Trinidad and Tobago',
                'United Arab Emirates','United Kingdom','United States','Uruguay'
            ],
            'iso_codes': [
                'AUS','AUT','BHR','BEL','CAN','CHL','HRV','CYP','CZE','DNK',
                'EST','FIN','FRA','DEU','GRC','HKG','HUN','ISL','IRL','ISR',
                'ITA','JPN','KOR','KWT','LVA','LTU','LUX','MLT','NLD','NZL',
                'NOR','OMN','PAN','POL','PRT','PRI','QAT','ROU','SAU','SGP',
                'SVK','SVN','ESP','SWE','CHE','TWN','TTO','ARE','GBR','USA','URY'
            ],
            'color': '#2E8B57'
        },

        'East Asia & Pacific (excluding high income)': {
            'countries': [
                'Cambodia','China','Indonesia','Lao PDR','Malaysia','Mongolia',
                'Myanmar','Philippines','Thailand','Viet Nam'
            ],
            'iso_codes': ['KHM','CHN','IDN','LAO','MYS','MNG','MMR','PHL','THA','VNM'],
            'color': '#FF6B35'
        },

        'Europe & Central Asia (excluding high income)': {
            'countries': [
                'Albania','Armenia','Azerbaijan','Belarus','Bosnia and Herzegovina','Bulgaria',
                'Georgia','Kazakhstan','Kosovo','Kyrgyz Republic','Moldova','Montenegro',
                'North Macedonia','Russian Federation','Serbia','Tajikistan','Turkiye',
                'Turkmenistan','Ukraine','Uzbekistan'
            ],
            'iso_codes': [
                'ALB','ARM','AZE','BLR','BIH','BGR','GEO','KAZ','XKX','KGZ',
                'MDA','MNE','MKD','RUS','SRB','TJK','TUR','TKM','UKR','UZB'
            ],
            'color': '#F7931E'
        },

        'South Asia (excluding high income)': {
            'countries': [
                'Afghanistan','Bangladesh','Bhutan','India','Maldives','Nepal','Pakistan','Sri Lanka'
            ],
            'iso_codes': ['AFG','BGD','BTN','IND','MDV','NPL','PAK','LKA'],
            'color': '#FFD23F'
        },

        'Latin America & Caribbean (excluding high income)': {
            'countries': [
                'Argentina','Belize','Bolivia','Brazil','Colombia','Costa Rica','Dominican Republic',
                'Ecuador','El Salvador','Guatemala','Haiti','Honduras','Jamaica','Mexico',
                'Nicaragua','Paraguay','Peru','Venezuela, RB'
            ],
            'iso_codes': [
                'ARG','BLZ','BOL','BRA','COL','CRI','DOM','ECU','SLV','GTM',
                'HTI','HND','JAM','MEX','NIC','PRY','PER','VEN'
            ],
            'color': '#E74C3C'
        },

        'Sub-Saharan Africa (excluding high income)': {
            'countries': [
                'Angola','Benin','Botswana','Burkina Faso','Burundi','Cameroon','Central African Republic',
                'Chad','Comoros','Congo, Dem. Rep.','Congo, Rep.','Cote d\'Ivoire','Eswatini','Ethiopia',
                'Gabon','Gambia, The','Ghana','Guinea','Kenya','Lesotho','Liberia','Madagascar','Malawi',
                'Mali','Mauritania','Mauritius','Mozambique','Namibia','Niger','Nigeria','Rwanda',
                'Senegal','Sierra Leone','Somalia','South Africa','South Sudan','Sudan','Tanzania',
                'Togo','Uganda','Zambia','Zimbabwe'
            ],
            'iso_codes': [
                'AGO','BEN','BWA','BFA','BDI','CMR','CAF','TCD','COM','COD','COG','CIV','SWZ','ETH',
                'GAB','GMB','GHA','GIN','KEN','LSO','LBR','MDG','MWI','MLI','MRT','MUS','MOZ','NAM',
                'NER','NGA','RWA','SEN','SLE','SOM','ZAF','SSD','SDN','TZA','TGO','UGA','ZMB','ZWE'
            ],
            'color': '#C0392B'
        },

        'Middle East & North Africa (excluding high income)': {
            'countries': [
                'Algeria','Djibouti','Egypt, Arab Rep.','Iran, Islamic Rep.','Iraq','Jordan','Lebanon',
                'Libya','Morocco','Syrian Arab Republic','Tunisia','West Bank and Gaza','Yemen, Rep.'
            ],
            'iso_codes': ['DZA','DJI','EGY','IRN','IRQ','JOR','LBN','LBY','MAR','SYR','TUN','PSE','YEM'],
            'color': '#8E44AD'
        }
    }

    # Enhanced regional mapping with comprehensive, data-driven recommendations
    region_mapping = {
        'High income': {
            'color': '#2E8B57',
            'countries': ['USA', 'Germany', 'Japan', 'UK', 'France', 'Canada', 'Australia'],
            'key_challenges': [
                'Digital divide among elderly populations (65+)',
                'Rural banking access gaps in remote areas',
                'Fintech regulation balancing innovation with consumer protection',
                'Financial literacy gaps despite high access rates'
            ],
            'opportunities': [
                'AI-driven personalized financial services',
                'Green finance and sustainable banking leadership',
                'Cross-border digital payments and open banking',
                'Advanced fraud detection and cybersecurity'
            ],
            'immediate_actions': [
                'Launch comprehensive digital literacy programs targeting 65+ demographic',
                'Deploy mobile banking units to underserved rural communities',
                'Establish regulatory sandboxes for fintech innovation testing',
                'Create senior-friendly banking interfaces and support systems'
            ],
            'medium_term': [
                'Implement AI-powered financial advisory services for mass market',
                'Develop comprehensive ESG (Environmental, Social, Governance) banking standards',
                'Build interoperable digital identity frameworks across countries',
                'Create inclusive design standards for all banking technologies'
            ],
            'long_term': [
                'Pioneer quantum-secure financial infrastructure',
                'Lead global financial inclusion measurement and reporting standards',
                'Achieve carbon-neutral banking operations by 2030',
                'Establish global fintech regulatory coordination mechanisms'
            ],
            'success_metrics': [
                'Achieve 95% digital banking adoption by 2030',
                'Ensure 100% rural area access within 10km radius',
                'Reduce elderly exclusion rate below 5%',
                'Maintain global leadership in financial innovation'
            ],
            'budget_allocation': {
                'Technology & Innovation': '40%',
                'Rural Infrastructure': '25%',
                'Elderly & Accessibility': '20%',
                'Sustainability Initiatives': '15%'
            }
        },
        
        'East Asia & Pacific (excluding high income)': {
            'color': '#FF6B35',
            'countries': ['China', 'Indonesia', 'Thailand', 'Philippines', 'Vietnam', 'Malaysia'],
            'key_challenges': [
                'Massive rural-urban digital divide (23% gap between urban and rural)',
                'Complex cross-border payment systems and currency fluctuations',
                'Regulatory fragmentation across diverse political systems',
                'Infrastructure gaps in remote island and mountain communities'
            ],
            'opportunities': [
                'World\'s largest mobile-first banking market potential',
                'E-commerce and digital marketplace integration',
                'Regional payment corridor development (ASEAN+3)',
                'Agricultural value chain financing innovation'
            ],
            'immediate_actions': [
                'Deploy rural 4G/5G infrastructure specifically for mobile banking access',
                'Launch unified regional QR code payment standards across ASEAN',
                'Establish cross-border fintech regulatory cooperation framework',
                'Create disaster-resilient payment systems for typhoon/earthquake zones'
            ],
            'medium_term': [
                'Build unified regional digital wallet ecosystem with currency conversion',
                'Develop agricultural value chain financing platforms for smallholder farmers',
                'Implement blockchain-based trade finance networks for SME exporters',
                'Create regional financial education programs in local languages'
            ],
            'long_term': [
                'Pioneer blockchain-based cross-border trade finance networks',
                'Lead ASEAN financial integration and common payment infrastructure',
                'Establish regional cryptocurrency and CBDC frameworks',
                'Achieve full financial inclusion in rural areas by 2035'
            ],
            'success_metrics': [
                'Reach 80% mobile payment adoption across all demographics',
                'Achieve 90% SME access to credit within 5 years',
                'Reduce remittance costs below $1 per $100 transferred',
                'Close rural-urban inclusion gap to under 10%'
            ],
            'budget_allocation': {
                'Mobile Infrastructure': '45%',
                'Rural & Agricultural Programs': '30%',
                'Regional Integration': '15%',
                'SME Financing': '10%'
            }
        },
        
        'Europe & Central Asia (excluding high income)': {
            'color': '#F7931E',
            'countries': ['Russia', 'Turkey', 'Kazakhstan', 'Ukraine', 'Romania', 'Bulgaria'],
            'key_challenges': [
                'Economic volatility and currency instability affecting savings behavior',
                'Legacy banking systems requiring modernization',
                'EU integration complexity for aspiring member states',
                'Post-conflict financial reconstruction in affected areas'
            ],
            'opportunities': [
                'Digital transformation acceleration post-COVID',
                'EU Single Euro Payments Area (SEPA) integration benefits',
                'Large diaspora remittance corridor optimization',
                'Energy sector transformation financing opportunities'
            ],
            'immediate_actions': [
                'Modernize core banking systems with cloud-based technology',
                'Launch currency-hedged savings products to protect against volatility',
                'Create EU payment directive compliance roadmaps for candidate countries',
                'Establish emergency financial services for conflict-affected populations'
            ],
            'medium_term': [
                'Build cross-border SME lending platforms leveraging EU programs',
                'Implement comprehensive open banking standards across the region',
                'Develop diaspora-focused investment and remittance products',
                'Create regional fintech hubs in major cities (Warsaw, Istanbul, Kyiv)'
            ],
            'long_term': [
                'Achieve full EU payment integration compliance for all candidate countries',
                'Pioneer post-conflict financial reconstruction and inclusion models',
                'Lead Eastern European fintech ecosystem development',
                'Establish regional financial stability mechanisms'
            ],
            'success_metrics': [
                'Achieve 70% digital banking adoption across all countries',
                'Complete EU payment integration for candidate countries by 2027',
                'Increase SME lending by 50% through cross-border platforms',
                'Reduce remittance costs to below 3% of transfer value'
            ],
            'budget_allocation': {
                'Banking System Modernization': '40%',
                'EU Integration Compliance': '25%',
                'SME & Entrepreneurship': '20%',
                'Post-Conflict Reconstruction': '15%'
            }
        },
        
        'South Asia (excluding high income)': {
            'color': '#FFD23F',
            'countries': ['India', 'Bangladesh', 'Pakistan', 'Sri Lanka', 'Nepal', 'Afghanistan'],
            'key_challenges': [
                'Massive unbanked population (400+ million adults)',
                'Significant gender inclusion gap (48% women vs 57% men)',
                'Documentation and identity verification barriers',
                'Low financial literacy rates, especially in rural areas'
            ],
            'opportunities': [
                'Digital identity systems success (Aadhaar model) replicable across region',
                'Mobile-first leapfrogging traditional banking infrastructure',
                'Government payment digitization and direct benefit transfers',
                'Massive youth population driving fintech adoption'
            ],
            'immediate_actions': [
                'Scale biometric-based account opening systems (replicating Aadhaar success)',
                'Establish women-only banking centers and female agent networks',
                'Digitize all government welfare and subsidy payment systems',
                'Launch multilingual financial literacy programs via mobile platforms'
            ],
            'medium_term': [
                'Build extensive agent banking networks reaching every village',
                'Develop alternative credit scoring using mobile phone and digital footprint data',
                'Implement blockchain-based land and property record systems',
                'Create region-wide instant payment systems with interoperability'
            ],
            'long_term': [
                'Achieve universal financial inclusion (80%+ account ownership) by 2030',
                'Lead global digital identity and inclusion standards development',
                'Pioneer climate-resilient agricultural finance and insurance systems',
                'Establish South Asian financial integration framework'
            ],
            'success_metrics': [
                'Reach 80% account ownership across all demographic groups',
                'Reduce gender gap in financial inclusion below 5%',
                'Deploy 1 million+ banking agents across rural areas',
                'Digitize 95% of government-to-person payments'
            ],
            'budget_allocation': {
                'Rural Infrastructure & Agents': '50%',
                'Women\'s Financial Inclusion': '25%',
                'Digital Identity Systems': '15%',
                'Agricultural Finance': '10%'
            }
        },
        
        'Latin America & Caribbean (excluding high income)': {
            'color': '#E74C3C',
            'countries': ['Brazil', 'Mexico', 'Argentina', 'Colombia', 'Peru', 'Chile'],
            'key_challenges': [
                'High informality rates (60%+ in some countries) limiting credit access',
                'Income inequality creating financial access barriers',
                'High remittance costs from USA (average 6.5% of transfer)',
                'Limited credit history and collateral for traditional banking'
            ],
            'opportunities': [
                'Rapidly growing fintech ecosystem and innovation',
                'Large diaspora creating remittance and investment flows',
                'Government service digitization and conditional cash transfers',
                'Strong mobile penetration enabling digital-first approaches'
            ],
            'immediate_actions': [
                'Launch alternative credit scoring systems for informal workers using utility and mobile data',
                'Create low-cost digital remittance corridors with USA and Spain',
                'Digitize all conditional cash transfer and social protection programs',
                'Establish fintech regulatory frameworks balancing innovation and protection'
            ],
            'medium_term': [
                'Build region-wide instant payment networks (building on PIX success)',
                'Develop comprehensive micro-insurance products for informal sector',
                'Create fintech accelerators and regulatory sandboxes in major cities',
                'Implement regional KYC and AML harmonization'
            ],
            'long_term': [
                'Pioneer AI-driven financial inclusion models for informal economies',
                'Lead regional economic integration through financial technology',
                'Achieve carbon-neutral payment systems and green finance leadership',
                'Establish Latin American fintech unicorn ecosystem'
            ],
            'success_metrics': [
                'Include 70% of informal sector workers in financial system',
                'Reduce remittance costs below $5 per $100 transferred',
                'Achieve 90% government payment digitization',
                'Create 50+ fintech unicorns by 2030'
            ],
            'budget_allocation': {
                'Fintech Ecosystem Development': '35%',
                'Informal Sector Inclusion': '30%',
                'Remittance Optimization': '20%',
                'Government Digitization': '15%'
            }
        },
        
        'Sub-Saharan Africa (excluding high income)': {
            'color': '#C0392B',
            'countries': ['Nigeria', 'Kenya', 'South Africa', 'Ghana', 'Tanzania', 'Ethiopia'],
            'key_challenges': [
                'Limited traditional banking infrastructure (1 branch per 20,000+ adults)',
                'Low and irregular income levels limiting savings capacity',
                'High transaction costs due to infrastructure limitations',
                'Limited electricity grid affecting digital payment systems'
            ],
            'opportunities': [
                'World-leading mobile money innovation and adoption',
                'Extensive agent banking network potential',
                'Agricultural value chain financing innovation opportunities',
                'Young, mobile-native population driving digital adoption'
            ],
            'immediate_actions': [
                'Expand mobile money interoperability across all countries and borders',
                'Train and deploy 500,000+ new banking agents in rural areas',
                'Install satellite-based internet infrastructure for remote banking access',
                'Launch solar-powered payment terminals and charging stations'
            ],
            'medium_term': [
                'Build comprehensive agricultural value chain financing platforms',
                'Create diaspora investment facilitation systems and platforms',
                'Develop climate-smart insurance and resilience products',
                'Establish regional mobile money and payment integration'
            ],
            'long_term': [
                'Lead global mobile money and agent banking innovation',
                'Pioneer space-based financial infrastructure for remote areas',
                'Achieve energy-independent banking and payment systems',
                'Create African Continental Free Trade Area payment integration'
            ],
            'success_metrics': [
                'Reach 60% mobile money usage across all demographics',
                'Ensure banking agent within 5km of every community',
                'Achieve 50% of smallholder farmers with access to credit',
                'Reduce transaction costs below 2% of transfer value'
            ],
            'budget_allocation': {
                'Mobile & Digital Infrastructure': '45%',
                'Agent Network Expansion': '25%',
                'Agricultural Finance': '20%',
                'Energy & Connectivity': '10%'
            }
        },
        
        'Middle East & North Africa (excluding high income)': {
            'color': '#8E44AD',
            'countries': ['Egypt', 'Morocco', 'Jordan', 'Tunisia', 'Algeria', 'Lebanon'],
            'key_challenges': [
                'Political instability affecting economic confidence and investment',
                'Youth unemployment rates (25%+ in many countries) limiting inclusion',
                'Regulatory restrictions on financial innovation and fintech',
                'Limited women\'s economic participation due to social and legal barriers'
            ],
            'opportunities': [
                'Islamic finance market growth and Sharia-compliant innovation',
                'Oil revenue diversification creating investment in financial inclusion',
                'Government digitization initiatives and e-government services',
                'Strategic location for Africa-Europe-Asia payment corridors'
            ],
            'immediate_actions': [
                'Launch comprehensive Sharia-compliant digital banking platforms',
                'Create dedicated youth entrepreneurship financing programs with lower requirements',
                'Digitize government salary, pension, and social service payments',
                'Establish women-only banking centers and female-focused financial products'
            ],
            'medium_term': [
                'Build regional Islamic fintech ecosystem and innovation hubs',
                'Develop sovereign wealth fund investments in financial inclusion technology',
                'Create post-conflict financial reconstruction and stability frameworks',
                'Implement region-wide Islamic banking and finance standards'
            ],
            'long_term': [
                'Lead global Islamic fintech innovation and standard-setting',
                'Pioneer oil-to-digital economy transition models for resource-rich countries',
                'Achieve regional financial market integration and payment systems',
                'Establish MENA as bridge for Africa-Europe-Asia financial flows'
            ],
            'success_metrics': [
                'Reach 50% Islamic finance adoption among Muslim populations',
                'Achieve 40% youth banking and financial service inclusion',
                'Digitize 80% of government payments and services',
                'Reduce gender gap in financial inclusion below 15%'
            ],
            'budget_allocation': {
                'Islamic Finance Innovation': '35%',
                'Youth Programs': '30%',
                'Government Digitization': '20%',
                'Women\'s Inclusion': '15%'
            }
        }
    }
    
    return pd.DataFrame(regional_data), pd.DataFrame(income_data), pd.DataFrame(feature_importance), region_mapping, country_mapping, regional_demographic_data

@st.cache_resource
def load_models():
    """Try to load models, fallback gracefully if not available"""
    try:
        regression_model = joblib.load('models/xgboost_regression_model.pkl')
        classification_model = joblib.load('models/lightgbm_classification_model.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return regression_model, classification_model, feature_names, True
    except:
        return None, None, None, False

def calculate_inclusion_score(inputs, region, income_group, models_loaded, regression_model, feature_names):
    """Calculate inclusion score using ML model if available, fallback to weighted calculation"""
    
    if models_loaded and regression_model is not None:
        try:
            # Map simplified inputs to full feature vector
            # Create feature vector with defaults and update with user inputs
            feature_vector = np.zeros(len(feature_names))
            
            # Map user inputs to corresponding model features
            feature_mapping = {
                'business_credit': ['business_loan_source', 'current_business_loans', 'credit_card_access', 'any_borrowing'],
                'digital_payments': ['mobile_money_transfers', 'digital_payment_usage', 'digital_payment_account', 'mobile_payment_services'],
                'active_savings': ['active_savings', 'savings_for_purchases', 'retirement_savings'],
                'mobile_banking': ['prefer_digital_finance', 'prefer_digital_accounts'],
                'emergency_funds': ['emergency_funds_available']
            }
            
            # Set feature values based on user inputs
            for input_key, feature_list in feature_mapping.items():
                user_value = inputs[input_key]
                for feature in feature_list:
                    if feature in feature_names:
                        idx = feature_names.index(feature)
                        feature_vector[idx] = user_value
            
            # Add regional and income encoding if features exist
            for i, feature in enumerate(feature_names):
                if region.lower().replace(' ', '_') in feature.lower():
                    feature_vector[i] = 1.0
                if income_group.lower().replace(' ', '_') in feature.lower():
                    feature_vector[i] = 1.0
            
            # Get model prediction
            prediction = regression_model.predict([feature_vector])[0]
            
            # Ensure prediction is in valid range
            final_score = max(0.05, min(0.95, prediction))
            
            return final_score
            
        except Exception as e:
            st.warning(f"Model prediction failed, using fallback calculation: {str(e)}")
    
    # Fallback calculation if model not available
    feature_weights = [0.6309, 0.5714, 0.4500, 0.5430, 0.4500]
    
    normalized_inputs = [
        inputs['business_credit'],
        inputs['digital_payments'],
        inputs['active_savings'],
        inputs['mobile_banking'],
        inputs['emergency_funds']
    ]
    
    weighted_score = sum(v * w for v, w in zip(normalized_inputs, feature_weights))
    base_score = weighted_score / sum(feature_weights)
    
    # Regional baseline adjustments
    regional_baselines = {
        'High income': 0.858,
        'East Asia & Pacific (excluding high income)': 0.568,
        'Europe & Central Asia (excluding high income)': 0.554,
        'South Asia (excluding high income)': 0.483,
        'Latin America & Caribbean (excluding high income)': 0.480,
        'Sub-Saharan Africa (excluding high income)': 0.427,
        'Middle East & North Africa (excluding high income)': 0.382
    }
    
    income_multipliers = {
        'High income': 1.1,
        'Upper middle income': 1.0,
        'Lower middle income': 0.9,
        'Low income': 0.8
    }
    
    regional_baseline = regional_baselines[region]
    income_multiplier = income_multipliers[income_group]
    
    final_score = (base_score * 0.7) + (regional_baseline * 0.3)
    final_score *= income_multiplier
    
    final_score = max(0.05, min(0.95, final_score))
    
    return final_score

def get_personalized_recommendations(score, inputs, region):
    """Generate personalized recommendations based on user profile"""
    recommendations = []
    
    # Business credit recommendations
    if inputs['business_credit'] < 0.3:
        recommendations.append({
            'title': '🏢 Build Business Credit History',
            'description': 'Start with microfinance or community lending programs',
            'priority': 'high'
        })
    
    # Digital payments
    if inputs['digital_payments'] < 0.4:
        recommendations.append({
            'title': '📱 Adopt Mobile Payment Solutions',
            'description': 'Begin with popular apps in your region for daily transactions',
            'priority': 'high'
        })
    
    # Savings behavior
    if inputs['active_savings'] < 0.5:
        recommendations.append({
            'title': '💰 Establish Regular Savings Habit',
            'description': 'Start small with automatic transfers or mobile savings apps',
            'priority': 'medium'
        })
    
    # Mobile banking
    if inputs['mobile_banking'] < 0.4:
        recommendations.append({
            'title': '🏦 Open Digital Banking Account',
            'description': 'Choose a bank with strong mobile features and low fees',
            'priority': 'high'
        })
    
    # Emergency funds
    if inputs['emergency_funds'] < 0.6:
        recommendations.append({
            'title': '🛡️ Build Emergency Fund',
            'description': 'Aim for 3-6 months of expenses in accessible savings',
            'priority': 'medium'
        })
    
    # Region-specific recommendations
    if region == 'Sub-Saharan Africa (excluding high income)':
        recommendations.append({
            'title': '📞 Leverage Mobile Money Services',
            'description': 'Use M-Pesa, MTN Mobile Money, or similar platforms',
            'priority': 'high'
        })
    elif region == 'South Asia (excluding high income)':
        recommendations.append({
            'title': '🆔 Utilize Digital Identity Systems',
            'description': 'Use Aadhaar or similar systems for easy account opening',
            'priority': 'medium'
        })
    
    return recommendations[:4]  # Return top 4 recommendations


# Streamlit App

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_region' not in st.session_state:
    st.session_state.selected_region = None

# Load data
regional_df, income_df, feature_df, region_mapping, country_mapping, regional_demographic_data = load_data()

# Header
st.markdown("""
<div class="main-header">
    <h1>🌍 FinScope Global</h1>
    <h2>Financial Inclusion Analytics Dashboard</h2>
    <p style="font-size: 18px; margin: 15px 0;"><i>AI-powered insights for evidence-based financial inclusion policy</i></p>
    <p style="font-size: 16px; font-weight: bold;">📊 149 countries | 🎯 89.6% ML accuracy | 🌐 8,311 adults analyzed</p>
</div>
""", unsafe_allow_html=True)

# Navigation
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("🏠 Dashboard Overview", key="home_btn"):
        st.session_state.page = 'home'
        st.session_state.selected_region = None

with col2:
    if st.button("🗺️ Regional Analytics", key="regional_btn"):
        st.session_state.page = 'regional'
        st.session_state.selected_region = None

with col3:
    if st.button("👤 Individual Analysis", key="individual_btn"):
        st.session_state.page = 'individual'

# Dashboard Overview Page
if st.session_state.page == 'home':
    st.markdown("## 📈 Global Financial Inclusion Overview")
    
    # Global Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin-top: 0;">🌐 Global Average</h3>
            <h1 style="color: #2d3436; margin: 15px 0; font-size: 3em;">61.1%</h1>
            <p style="color: #636e72; font-size: 16px;">Financial Inclusion Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin-top: 0;">🏆 Best Performing</h3>
            <h1 style="color: #2E8B57; margin: 15px 0; font-size: 3em;">85.8%</h1>
            <p style="color: #636e72; font-size: 16px;">High Income Countries</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin-top: 0;">🎯 Largest Gap</h3>
            <h1 style="color: #8E44AD; margin: 15px 0; font-size: 3em;">47.6%</h1>
            <p style="color: #636e72; font-size: 16px;">MENA vs High Income</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin-top: 0;">🤖 ML Accuracy</h3>
            <h1 style="color: #2d3436; margin: 15px 0; font-size: 3em;">89.6%</h1>
            <p style="color: #636e72; font-size: 16px;">Random Forest Model</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Regional Comparison
    st.markdown("### 🌍 Regional Performance at a Glance")
    
    fig_overview = px.bar(
        regional_df.sort_values('inclusion_rate', ascending=True),
        x='inclusion_rate',
        y='region',
        orientation='h',
        color='inclusion_rate',
        color_continuous_scale=[
            [0.0, '#8E44AD'],  # MENA
            [0.2, '#C0392B'],  # Sub-Saharan Africa
            [0.4, '#E74C3C'],  # Latin America
            [0.5, '#FFD23F'],  # South Asia
            [0.6, '#F7931E'],  # Europe Central Asia
            [0.8, '#FF6B35'],  # East Asia Pacific
            [1.0, '#2E8B57']   # High income
        ],
        text='inclusion_rate',
        title="<b>Financial Inclusion Rates by Region</b>",
        height=500
    )
    
    fig_overview.update_traces(
        texttemplate='%{text:.1%}', 
        textposition='outside',
        textfont=dict(size=14, color='black', family='Arial Black')
    )
    
    fig_overview.update_layout(
        xaxis_title="Financial Inclusion Rate",
        yaxis_title="",
        xaxis=dict(tickformat='.0%'),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12), 
        title_font=dict(size=18),
        yaxis=dict(title_font_size=12, tickfont_size=11)
    )
    
    st.plotly_chart(fig_overview, use_container_width=True)
    
       # Income Group Analysis
    st.markdown("### 🎯 Income Group Analysis")
    
    fig_income = px.bar(
        income_df,
        x='income_group',
        y='inclusion_rate',
        color='inclusion_rate',
        color_continuous_scale='Viridis',
        text='inclusion_rate',
        title="<b>Financial Inclusion by Income Level</b>",
        height=400
    )
    
    fig_income.update_traces(
        texttemplate='%{text:.1%}',
        textposition='outside',
        textfont=dict(size=14, color='black', family='Arial Black')
    )
    
    fig_income.update_layout(
        showlegend=False,
        xaxis_title="Income Group",
        yaxis_title="Inclusion Rate",
        yaxis=dict(tickformat='.0%'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12), 
        title_font=dict(size=16)
    )
    
    fig_income1 = px.pie(
        income_df,
        names='income_group',
        values='inclusion_rate',
        color_discrete_sequence=px.colors.sequential.Viridis,
        title="<b>Inclusion Rate Distribution by Income Group</b>",
        height=400
    )   
    fig_income1.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14)
    fig_income1.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12), 
        title_font=dict(size=16)
    )       
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_income1, use_container_width=True)
    with col2:          
        st.plotly_chart(fig_income, use_container_width=True)

    
    
    # Enhanced Demographic Analysis with Regional Range Comparison
    st.markdown("### 🌍 Global Financial Inclusion: Demographic Range Analysis")
    st.markdown("*Complete range across all regions for each demographic group*")

    # Prepare data for range chart
    demographic_groups = []
    min_values = []
    max_values = []
    ranges = []
    best_regions = []
    worst_regions = []

    for demo_group, regions in regional_demographic_data.items():
        values = list(regions.values())
        region_names = list(regions.keys())
        
        min_val = min(values)
        max_val = max(values)
        
        demographic_groups.append(demo_group)
        min_values.append(min_val)
        max_values.append(max_val)
        ranges.append(max_val - min_val)
        
        best_regions.append(region_names[values.index(max_val)])
        worst_regions.append(region_names[values.index(min_val)])

    # Regional performance heatmap
    st.markdown("### 🔥 Regional Performance Heatmap")
    st.markdown("*Color-coded performance across all demographic groups by region*")

    # Create heatmap data
    heatmap_data = []
    regions_list = ['High income', 'Europe & Central Asia (excluding high income)', 'East Asia & Pacific (excluding high income)', 'South Asia (excluding high income)', 'Latin America & Caribbean (excluding high income)', 'Middle East & North Africa (excluding high income)', 'Sub-Saharan Africa (excluding high income)']

    for demo_group in demographic_groups:
        row = []
        for region in regions_list:
            if region in regional_demographic_data[demo_group]:
                row.append(regional_demographic_data[demo_group][region])
            else:
                row.append(None)
        heatmap_data.append(row)



    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[region.replace(' (excluding high income)', '').replace(' & ', ' &<br>') for region in regions_list],
        y=demographic_groups,
        colorscale='RdYlGn',
        zmin=0,
        zmax=1,
        colorbar=dict(
            title="Account<br>Ownership<br>Rate",
            tickmode="linear",
            tick0=0,
            dtick=0.2,
            tickformat='.0%'
        ),
        hoverongaps=False,
        hovertemplate="<b>%{y}</b><br>%{x}<br>Rate: %{z:.0%}<extra></extra>"
    ))

    fig_heatmap.update_layout(
        title=dict(
            text="<b>Regional Performance Matrix</b><br><span style='font-size:14px; color:#666'>Account ownership rates by demographic group and region</span>",
            x=0.5,
            font=dict(size=18)
        ),
        height=600,
        font=dict(family="Arial, sans-serif", size=11),
        margin=dict(l=150, r=100, t=80, b=100)
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)



    # Create range chart
    fig_range = go.Figure()

    # Sort by range size (largest disparities first)
    sorted_indices = sorted(range(len(ranges)), key=lambda i: ranges[i], reverse=True)

    y_labels_sorted = [demographic_groups[i] for i in sorted_indices]
    min_vals_sorted = [min_values[i] for i in sorted_indices]
    max_vals_sorted = [max_values[i] for i in sorted_indices]
    ranges_sorted = [ranges[i] for i in sorted_indices]
    best_regions_sorted = [best_regions[i] for i in sorted_indices]
    worst_regions_sorted = [worst_regions[i] for i in sorted_indices]

    # Add range bars (from min to max)
    for i in range(len(y_labels_sorted)):
        # Background range bar
        fig_range.add_trace(go.Bar(
            name='Regional Range' if i == 0 else '',
            y=[y_labels_sorted[i]],
            x=[max_vals_sorted[i] - min_vals_sorted[i]],
            base=[min_vals_sorted[i]],
            orientation='h',
            marker=dict(color='lightgray', opacity=0.5),
            showlegend=True if i == 0 else False,
            hovertemplate=f"<b>{y_labels_sorted[i]}</b><br>Range: {ranges_sorted[i]:.0%}<br>Best: {best_regions_sorted[i]} ({max_vals_sorted[i]:.0%})<br>Worst: {worst_regions_sorted[i]} ({min_vals_sorted[i]:.0%})<extra></extra>"
        ))
        
        # Min value marker
        fig_range.add_trace(go.Scatter(
            name='Lowest Rate' if i == 0 else '',
            x=[min_vals_sorted[i]],
            y=[y_labels_sorted[i]],
            mode='markers',
            marker=dict(color='#E74C3C', size=12, symbol='circle'),
            showlegend=True if i == 0 else False,
            hovertemplate=f"<b>Lowest:</b> {worst_regions_sorted[i]}<br>Rate: {min_vals_sorted[i]:.0%}<extra></extra>"
        ))
        
        # Max value marker
        fig_range.add_trace(go.Scatter(
            name='Highest Rate' if i == 0 else '',
            x=[max_vals_sorted[i]],
            y=[y_labels_sorted[i]],
            mode='markers',
            marker=dict(color='#27AE60', size=12, symbol='circle'),
            showlegend=True if i == 0 else False,
            hovertemplate=f"<b>Highest:</b> {best_regions_sorted[i]}<br>Rate: {max_vals_sorted[i]:.0%}<extra></extra>"
        ))

    # Add range annotations
    for i in range(len(y_labels_sorted)):
        fig_range.add_annotation(
            x=(min_vals_sorted[i] + max_vals_sorted[i]) / 2,
            y=i,
            text=f"{ranges_sorted[i]:.0%}",
            showarrow=False,
            font=dict(color='black', size=11, family='Arial Black'),
            bgcolor='white',
            bordercolor='gray',
            borderwidth=1
        )

    fig_range.update_layout(
        title=dict(
            text="<b>Global Financial Inclusion Disparities</b><br><span style='font-size:14px; color:#666'>Regional range for each demographic group (sorted by disparity size)</span>",
            x=0.5,
            font=dict(size=18)
        ),
        xaxis=dict(
            title="Account Ownership Rate",
            tickformat='.0%',
            range=[0, 1.0],
            gridcolor='lightgray',
            gridwidth=1
        ),
        yaxis=dict(title="Demographic Groups"),
        height=700,
        font=dict(family="Arial, sans-serif", size=11),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=150, r=50, t=100, b=50)
    )

    st.plotly_chart(fig_range, use_container_width=True)


    # Summary insights
    st.markdown("### 📊 Key Disparities Revealed")

    col1, col2, col3, col4 = st.columns(4)

    largest_disparity_idx = ranges.index(max(ranges))
    smallest_disparity_idx = ranges.index(min(ranges))

    with col1:
        st.metric(
            label="🔴 Largest Disparity",
            value=f"{max(ranges):.0%}",
            delta=f"{demographic_groups[largest_disparity_idx]}"
        )

    with col2:
        st.metric(
            label="🟢 Smallest Disparity",
            value=f"{min(ranges):.0%}",
            delta=f"{demographic_groups[smallest_disparity_idx]}"
        )

    with col3:
        best_overall_region = max(regions_list, key=lambda r: np.mean([regional_demographic_data[d].get(r, 0) for d in demographic_groups]))
        best_avg = np.mean([regional_demographic_data[d].get(best_overall_region, 0) for d in demographic_groups])
        st.metric(
            label="🏆 Best Overall Region",
            value=f"{best_avg:.0%}",
            delta=best_overall_region.replace(' (excluding high income)', '')
        )

    with col4:
        worst_overall_region = min(regions_list, key=lambda r: np.mean([regional_demographic_data[d].get(r, 0) for d in demographic_groups]))
        worst_avg = np.mean([regional_demographic_data[d].get(worst_overall_region, 0) for d in demographic_groups])
        st.metric(
            label="🎯 Most Needs Support",
            value=f"{worst_avg:.0%}",
            delta=worst_overall_region.replace(' (excluding high income)', '')
        )

    # Action insights
    st.markdown("### 🚀 Strategic Insights")

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin: 10px 0;">
    <h4 style="margin-top: 0; color: white;">Critical Findings:</h4>
    <ul>
    <li><strong>Highest Disparity:</strong> {demographic_groups[largest_disparity_idx]} shows {max(ranges):.0%} gap between regions</li>
    <li><strong>Most Consistent:</strong> {demographic_groups[smallest_disparity_idx]} has only {min(ranges):.0%} variation across regions</li>
    <li><strong>Universal Challenge:</strong> All demographic groups show significant regional variations</li>
    <li><strong>High Income Dominance:</strong> Consistently leads in nearly all categories</li>
    <li><strong>MENA & Sub-Saharan Africa:</strong> Most frequently appear as lowest performers</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
        
# Regional Analytics Page
elif st.session_state.page == 'regional':
    st.markdown("## 🗺️ Interactive Regional Analytics")
    st.markdown("### *Click on any region to explore detailed insights and recommendations*")
    
    # Create choropleth map with actual countries
    country_data = []
    for region, region_info in country_mapping.items():
        region_rate = regional_df[regional_df['region'] == region]['inclusion_rate'].iloc[0]
        region_count = regional_df[regional_df['region'] == region]['count'].iloc[0]
        
        for i, (country, iso_code) in enumerate(zip(region_info['countries'], region_info['iso_codes'])):
            country_data.append({
                'country': country,
                'iso_code': iso_code,
                'region': region,
                'inclusion_rate': region_rate,
                'color': region_info['color'],
                'sample_size': region_count
            })

    country_df = pd.DataFrame(country_data)

    # Create the choropleth map
    fig_choropleth = go.Figure(data=go.Choropleth(
        locations=country_df['iso_code'],
        z=country_df['inclusion_rate'],
        locationmode='ISO-3',
        colorscale=[
            [0.0, '#8E44AD'],    # MENA
            [0.15, '#C0392B'],   # Sub-Saharan Africa  
            [0.30, '#E74C3C'],   # Latin America
            [0.45, '#FFD23F'],   # South Asia
            [0.60, '#F7931E'],   # Europe & Central Asia
            [0.75, '#FF6B35'],   # East Asia Pacific
            [1.0, '#2E8B57']     # High income
        ],
        text=country_df['country'],
        hovertemplate='<b>%{text}</b><br>' +
                      'Region: %{customdata[0]}<br>' +
                      'Inclusion Rate: %{z:.1%}<br>' +
                      'Sample Size: %{customdata[1]:,}<br>' +
                      '<extra></extra>',
        customdata=country_df[['region', 'sample_size']].values,
        colorbar=dict(
            title=dict(
                text="Financial<br>Inclusion Rate",
                font=dict(size=14)
            ),
            tickformat='.0%',
            len=0.8
        ),
        showscale=True
    ))

    fig_choropleth.update_layout(
        title={
            'text': '<b>Global Financial Inclusion Rates by Country</b><br><sub>Countries color-coded by regional inclusion rates - Click to explore regions</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=18)
        },
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="rgb(180,180,180)",
            projection_type='natural earth',
            bgcolor='rgba(240,240,240,0.1)',
            showland=True,
            landcolor='rgb(250,250,250)',
            showocean=True,
            oceancolor='rgb(230,245,255)',
            showlakes=True,
            lakecolor='rgb(230,245,255)'
        ),
        height=600,
        margin=dict(l=0, r=0, t=80, b=0)
    )

    # Display the interactive map
    st.plotly_chart(fig_choropleth, use_container_width=True, key="country_choropleth")
    
    # Region selection buttons
    st.markdown("### 📊 Select Region for Detailed Analysis")
    
    regions_sorted = regional_df.sort_values('inclusion_rate', ascending=False)
    cols = st.columns(2)
    
    for idx, (_, region_data) in enumerate(regions_sorted.iterrows()):
        region_name = region_data['region']
        inclusion_rate = region_data['inclusion_rate']
        region_color = country_mapping[region_name]['color']
        
        col = cols[idx % 2]
        with col:
            if st.button(f"{region_name.split('(')[0].strip()} - {inclusion_rate:.1%}", 
                        key=f"region_{idx}"):
                st.session_state.selected_region = region_name
    
    
    
    # Display selected region details with enhanced recommendations
if st.session_state.selected_region:
    region_name = st.session_state.selected_region
    region_data = regional_df[regional_df['region'] == region_name].iloc[0]
    region_info = region_mapping[region_name]
    
    # Get priority demographics for this region
    region_demographics = regional_demographic_data
    priority_groups = []
    
    # Define conditions to ignore certain demographic groups
    def should_ignore_demo_group(demo_group, rate, region_name=None):
        """
        Define conditions under which a demographic group should be ignored
        Add your specific conditions here
        """
        # Example conditions - modify based on your requirements:
        
        # Ignore if inclusion rate is already very high (above 80%)
        if rate > 0.8:
            return True
        
        # Ignore specific demographics in certain regions
        if region_name:
            # Example: Ignore rural demographics in highly urbanized regions
            if 'Rural' in demo_group and region_name in ['Europe & Central Asia', 'North America']:
                return True
            
            # Example: Ignore certain age groups in specific contexts
            if 'Age 65+' in demo_group and rate > 0.6:
                return True
        
        # Ignore if demographic data quality is poor (you can add data quality checks)
        # if has_poor_data_quality(demo_group, region_name):
        #     return True
        
        # Add more conditions as needed
        return False
    
    # Identify groups with lowest inclusion rates in this region
    for demo_group, regions in region_demographics.items():
        if region_name in regions:
            rate = regions[region_name]
            
            # Apply conditional logic - skip if conditions are met
            if should_ignore_demo_group(demo_group, rate, region_name):
                continue  # Skip this demographic group
            
            if rate < 0.5:  # Below 50% inclusion
                priority_groups.append((demo_group, rate))
    
    # Sort by lowest rates first
    priority_groups.sort(key=lambda x: x[1])
    
    st.markdown(f"""
    <div class="region-info-card" style="border-left-color: {region_info['color']};">
        <h2 style="color: {region_info['color']}; margin-top: 0;">
            {region_name.split('(')[0].strip()} - Strategic Analysis & Priority Demographics
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Priority Demographics Alert - only show if we have valid priority groups
    if priority_groups:
        st.markdown("#### 🚨 HIGHEST PRIORITY DEMOGRAPHICS")
        st.markdown("*These groups require immediate, targeted interventions*")
        
        priority_cols = st.columns(min(3, len(priority_groups)))
        for idx, (group, rate) in enumerate(priority_groups[:3]):
            with priority_cols[idx]:
                st.markdown(f"""
                <div class="demo-card priority-card">
                    <h4 style="color: #E74C3C; margin: 0 0 10px 0;">⚠️ CRITICAL</h4>
                    <h3 style="margin: 0 0 5px 0;">{group}</h3>
                    <h2 style="color: #E74C3C; margin: 5px 0;">{rate:.0%}</h2>
                    <p style="margin: 5px 0; font-size: 12px; color: #666;">Inclusion Rate</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        # Show message when no priority demographics meet criteria
        st.markdown("#### ✅ DEMOGRAPHIC INCLUSION STATUS")
        st.markdown("*All demographic groups in this region meet inclusion thresholds or have been addressed*")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Inclusion Rate", f"{region_data['inclusion_rate']:.1%}")
    with col2:
        st.metric("Sample Size", f"{region_data['count']:,}")
    with col3:
        gap_to_best = regional_df['inclusion_rate'].max() - region_data['inclusion_rate']
        st.metric("Gap to Best", f"{gap_to_best:.1%}")
    with col4:
        rank = (regional_df['inclusion_rate'] > region_data['inclusion_rate']).sum() + 1
        st.metric("Global Rank", f"#{rank}/7")
    
    # Only show recommendations section if we have valid priority groups
    if priority_groups:
        # Generate specific, demographic-targeted recommendations
        st.markdown(f"""
        <div class="recommendation-section" style="border-left-color: {region_info['color']};">
            <h3 style="color: {region_info['color']}; margin-top: 0;">🎯 Personalized Action Plan for Priority Demographics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create targeted recommendations based on actual demographic data
        demographic_actions = {}
        
        # Generate specific actions for each priority demographic
        for demo_group, rate in priority_groups:
            # Skip if this group should be ignored (double-check)
            if should_ignore_demo_group(demo_group, rate, region_name):
                continue
                
            actions = []
            
            if 'Women' in demo_group and rate < 0.5:
                actions.extend([
                    f"📱 Deploy 10,000+ female banking agents within 6 months - current women's inclusion at {rate:.0%}",
                    f"🏦 Establish women-only banking hours (7-10 AM) in all branches to address cultural barriers",
                    f"💳 Launch women-specific savings accounts with no minimum balance - targeting {rate:.0%} gap closure",
                    f"👥 Create female-led financial literacy circles in communities with <40% women inclusion"
                ])
            
            if 'Poor' in demo_group and rate < 0.4:
                actions.extend([
                    f"💰 Launch $10 minimum account opening for bottom 40% income group (currently {rate:.0%})",
                    f"📊 Deploy income-graduated fee structures - free transactions for <$2/day earners",
                    f"🌾 Create agricultural value chain financing for 50,000+ smallholder farmers",
                    f"💸 Offer micro-savings products with daily deposit options starting at $0.25"
                ])
            
            if 'Rural' in demo_group and rate < 0.6:
                actions.extend([
                    f"🚐 Deploy mobile banking units to 500+ villages with <50% banking access",
                    f"📡 Install satellite-powered banking kiosks in areas >20km from nearest branch",
                    f"👨‍🌾 Train 5,000+ rural agents in agricultural communities within 12 months",
                    f"☀️ Establish solar-powered payment terminals in off-grid rural areas"
                ])
            
            if 'Primary Education' in demo_group and rate < 0.4:
                actions.extend([
                    f"📚 Launch visual/audio banking tutorials in local languages for low-literacy populations",
                    f"🤝 Partner with adult literacy programs to integrate basic financial education",
                    f"📱 Deploy voice-activated banking systems for non-literate users",
                    f"👥 Create peer-to-peer learning networks in communities with <50% literacy"
                ])
            
            if 'Age 15-24' in demo_group and rate < 0.5:
                actions.extend([
                    f"🎮 Launch gamified savings apps targeting {rate:.0%} of unbanked youth",
                    f"🎓 Partner with schools/universities for mandatory financial literacy courses",
                    f"💼 Create youth entrepreneur loan programs with simplified documentation",
                    f"📲 Deploy social media-based financial education campaigns in local languages"
                ])
            
            if 'Out of Labor Force' in demo_group and rate < 0.5:
                actions.extend([
                    f"👵 Target elderly and retired populations with simplified banking products",
                    f"🤱 Create childcare-linked savings accounts for stay-at-home parents",
                    f"🏥 Partner with healthcare providers for health savings account integration",
                    f"💻 Develop disability-accessible banking interfaces and services"
                ])
            
            if actions:  # Only add if we have actions for this group
                demographic_actions[demo_group] = actions[:5]  # Top 4 most relevant actions
        
        # Display demographic-specific action plans - only if we have valid actions
        if demographic_actions:
            st.markdown("#### 🚀 IMMEDIATE ACTIONS BY PRIORITY DEMOGRAPHIC")
            
            for demo_group, actions in list(demographic_actions.items())[:3]:  # Show top 3 priority groups
                rate = next(r for g, r in priority_groups if g == demo_group)
                st.markdown(f"""
                <div class="action-card immediate" style="margin: 15px 0;">
                    <h5 style="margin: 0 0 10px 0; color: #E74C3C;">🎯 TARGET: {demo_group} ({rate:.0%} inclusion)</h5>
                    <ul style="margin: 5px 0; padding-left: 20px;">
                        {''.join(f'<li style="margin: 3px 0;">{action}</li>' for action in actions)}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        # Calculate demographic impact potential - only if we have priority groups
        if priority_groups:
            st.markdown("#### 📊 DEMOGRAPHIC IMPACT CALCULATOR")
            
            # Calculate potential reach for each intervention
            impact_data = []
            for demo_group, current_rate in priority_groups[:3]:
                # Skip if this group should be ignored
                if should_ignore_demo_group(demo_group, current_rate, region_name):
                    continue
                    
                target_rate = min(current_rate + 0.25, 0.8)  # Realistic 25% improvement or 80% cap
                potential_newly_included = (target_rate - current_rate) * region_data['count']
                
                impact_data.append({
                    'Demographic': demo_group,
                    'Current Rate': f"{current_rate:.0%}",
                    'Target Rate': f"{target_rate:.0%}",
                    'Potential New Users': f"{potential_newly_included:,.0f}",
                    'Investment Priority': '🔴 HIGH' if current_rate < 0.3 else '🟡 MEDIUM'
                })
            
            if impact_data:  # Only show table if we have data
                impact_df = pd.DataFrame(impact_data)
                st.dataframe(impact_df, use_container_width=True, hide_index=True)
        
        # Resource allocation calculator - only if we have priority groups
        if priority_groups:
            st.markdown("#### 💰 TARGETED BUDGET ALLOCATION")
            
            total_unbanked = region_data['count'] * (1 - region_data['inclusion_rate'])
            allocation_data = []
            
            for demo_group, rate in priority_groups[:4]:
                # Skip if this group should be ignored
                if should_ignore_demo_group(demo_group, rate, region_name):
                    continue
                    
                # Estimate demographic size as proportion of total unbanked
                if 'Women' in demo_group:
                    demo_unbanked = total_unbanked * 0.52  # Approx 52% women globally
                elif 'Rural' in demo_group:
                    demo_unbanked = total_unbanked * 0.6   # Approx 60% rural in developing regions
                elif 'Poor' in demo_group:
                    demo_unbanked = total_unbanked * 0.4   # Bottom 40%
                elif 'Youth' in demo_group:
                    demo_unbanked = total_unbanked * 0.25  # Approx 25% youth
                else:
                    demo_unbanked = total_unbanked * 0.3   # Default estimate
                
                # Calculate cost per inclusion (varies by demographic complexity)
                if rate < 0.3:
                    cost_per_person = 50  # High-touch interventions needed
                elif rate < 0.5:
                    cost_per_person = 30  # Moderate interventions
                else:
                    cost_per_person = 20  # Standard interventions
                
                total_investment = demo_unbanked * cost_per_person
                
                allocation_data.append({
                    'Priority Group': demo_group,
                    'Unbanked Population': f"{demo_unbanked:,.0f}",
                    'Cost per Person': f"${cost_per_person}",
                    'Total Investment': f"${total_investment:,.0f}",
                    'Expected Timeline': '18-36 months'
                })
            
            if allocation_data:  # Only show if we have valid allocation data
                allocation_df = pd.DataFrame(allocation_data)
                st.dataframe(allocation_df, use_container_width=True, hide_index=True)
                
                total_investment = sum(float(row['Total Investment'].replace('$', '').replace(',', '')) for _, row in allocation_df.iterrows())
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #E74C3C, #C0392B); padding: 15px; border-radius: 10px; color: white; margin: 15px 0;">
                    <h4 style="margin: 0; color: white;">💡 TOTAL PRIORITY INVESTMENT NEEDED</h4>
                    <h2 style="margin: 10px 0; color: white;">${total_investment:,.0f}</h2>
                    <p style="margin: 0;">To target the most excluded demographics with evidence-based interventions</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Action Timeline (always show regional actions)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🚀 Immediate Actions (0-12 months)")
        for action in region_info['immediate_actions']:
            st.markdown(f"""
            <div class="action-card immediate">
                <h5 style="margin: 0 0 10px 0; color: #E74C3C;">⚡ Priority Action</h5>
                <p style="margin: 0; font-size: 14px;">{action}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🔄 Medium-term (1-3 years)")
        for action in region_info['medium_term']:
            st.markdown(f"""
            <div class="action-card medium-term">
                <h5 style="margin: 0 0 10px 0; color: #F39C12;">🔧 Strategic Initiative</h5>
                <p style="margin: 0; font-size: 14px;">{action}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("#### 🌟 Long-term Vision (3-10 years)")
        for action in region_info['long_term']:
            st.markdown(f"""
            <div class="action-card long-term">
                <h5 style="margin: 0 0 10px 0; color: #27AE60;">🎯 Transformation Goal</h5>
                <p style="margin: 0; font-size: 14px;">{action}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # KPIs and Budget Allocation (always show)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-section">
            <h4 style="margin-top: 0;">📊 Success Metrics & KPIs</h4>
            <ul style="margin: 10px 0; padding-left: 20px;">
                {''.join(f'<li style="margin: 5px 0;">{metric}</li>' for metric in region_info['success_metrics'])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-section">
            <h4 style="margin-top: 0;">💰 Recommended Budget Allocation</h4>
            <ul style="margin: 10px 0; padding-left: 20px;">
                {''.join(f'<li style="margin: 5px 0;"><strong>{area}:</strong> {percentage}</li>' for area, percentage in region_info['budget_allocation'].items())}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Challenges and Opportunities (always show)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚧 Key Challenges")
        for challenge in region_info['key_challenges']:
            st.markdown(f"• {challenge}")
    
    with col2:
        st.markdown("#### 🚀 Growth Opportunities")
        for opportunity in region_info['opportunities']:
            st.markdown(f"• {opportunity}")
  
# Individual Analysis Page (Simplified)
elif st.session_state.page == 'individual':
    st.markdown("## 👤 Personal Financial Inclusion Assessment")
    
    if models_loaded:
        st.markdown("*🤖 **AI-Powered Prediction**: Using XGBoost ML model (88.5% R²) for precise analysis*")
    else:
        st.markdown("*📊 **Statistical Analysis**: Answer 5 simple questions for personalized insights*")
    
    # Show model status prominently
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 📝 Quick Assessment (5 Questions)")
    with col2:
        if models_loaded:
            st.success("🤖 ML Model Active")
        else:
            st.warning("📊 Fallback Mode")
    
    # Simplified input form with just 5 key questions
    with st.form("simplified_assessment"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📍 Location & Income**")
            region = st.selectbox("Your Region", [
                'High income',
                'East Asia & Pacific (excluding high income)',
                'Europe & Central Asia (excluding high income)',
                'South Asia (excluding high income)',
                'Latin America & Caribbean (excluding high income)',
                'Sub-Saharan Africa (excluding high income)',
                'Middle East & North Africa (excluding high income)'
            ], help="Select your geographic region")
            
            income_group = st.selectbox("Income Level", [
                'High income', 'Upper middle income', 'Lower middle income', 'Low income'
            ], help="Choose your income bracket")
            
            st.markdown("**🏢 Credit & Business**")
            business_credit = st.slider("Business/Credit Access", 1, 10, 5,
                                      help="How easily can you get loans or credit?") / 10.0
            
        with col2:
            st.markdown("**📱 Digital Banking**")
            digital_payments = st.slider("Digital Payment Usage", 1, 10, 5,
                                        help="How often do you use mobile/online payments?") / 10.0
            
            mobile_banking = st.slider("Mobile Banking Activity", 1, 10, 5,
                                     help="How actively do you use banking apps?") / 10.0
            
            st.markdown("**💰 Savings & Emergency**")
            active_savings = st.slider("Regular Savings", 1, 10, 5,
                                     help="How consistently do you save money?") / 10.0
            
            emergency_funds = st.slider("Emergency Preparedness", 1, 10, 5,
                                      help="Do you have funds for emergencies?") / 10.0
        
        submitted = st.form_submit_button("🤖 Get AI Analysis", 
                                        type="primary", use_container_width=True)
    
    if submitted:
        # Prepare inputs
        user_inputs = {
            'business_credit': business_credit,
            'digital_payments': digital_payments,
            'active_savings': active_savings,
            'mobile_banking': mobile_banking,
            'emergency_funds': emergency_funds
        }
        
        # Calculate inclusion score using ML model if available
        inclusion_score = calculate_inclusion_score(user_inputs, region, income_group, 
                                                   models_loaded, regression_model, feature_names)
        
        # Show model status
        if models_loaded:
            st.success("✅ Using advanced ML model (XGBoost) for prediction")
        else:
            st.info("ℹ️ Using statistical model - install ML libraries for enhanced accuracy")
        
        # Determine status
        if inclusion_score >= 0.75:
            status_color, status_emoji, status_text = "#27AE60", "✅", "Financially Well-Included"
            status_desc = "You have strong financial inclusion with good access to services."
        elif inclusion_score >= 0.60:
            status_color, status_emoji, status_text = "#F39C12", "⚠️", "Moderately Included"
            status_desc = "You have decent financial access but room for improvement."
        elif inclusion_score >= 0.40:
            status_color, status_emoji, status_text = "#FF6B35", "⚠️", "Partially Included"
            status_desc = "You have basic financial access but significant gaps remain."
        else:
            status_color, status_emoji, status_text = "#E74C3C", "🚨", "Limited Financial Access"
            status_desc = "You face major barriers to financial services and need support."
        
        # Display results
        st.markdown(f"""
        <div class="score-card" style="border-left: 6px solid {status_color};">
            <h2 style="color: {status_color}; margin-top: 0;">{status_emoji} {status_text}</h2>
            <h1 style="font-size: 3em; margin: 15px 0; color: {status_color};">{inclusion_score:.0%}</h1>
            <p style="font-size: 1.1em; color: #636e72; margin-bottom: 0;">{status_desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Comparison with regional average
        regional_avg = regional_df[regional_df['region'] == region]['inclusion_rate'].iloc[0]
        income_avg = income_df[income_df['income_group'] == income_group]['inclusion_rate'].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            diff_regional = inclusion_score - regional_avg
            color = "#27AE60" if diff_regional > 0 else "#E74C3C"
            arrow = "↗️" if diff_regional > 0 else "↘️"
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #667eea; margin-top: 0;">vs Regional Average</h4>
                <h2 style="color: {color}; margin: 10px 0;">{arrow} {diff_regional:+.1%}</h2>
                <p style="color: #636e72;">Regional: {regional_avg:.0%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            diff_income = inclusion_score - income_avg
            color = "#27AE60" if diff_income > 0 else "#E74C3C"
            arrow = "↗️" if diff_income > 0 else "↘️"
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #667eea; margin-top: 0;">vs Income Group</h4>
                <h2 style="color: {color}; margin: 10px 0;">{arrow} {diff_income:+.1%}</h2>
                <p style="color: #636e72;">Income Group: {income_avg:.0%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            global_avg = 0.581
            diff_global = inclusion_score - global_avg
            color = "#27AE60" if diff_global > 0 else "#E74C3C"
            arrow = "↗️" if diff_global > 0 else "↘️"
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #667eea; margin-top: 0;">vs Global Average</h4>
                <h2 style="color: {color}; margin: 10px 0;">{arrow} {diff_global:+.1%}</h2>
                <p style="color: #636e72;">Global: {global_avg:.0%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature analysis radar chart
        st.markdown("### 📊 Your Financial Profile")
        
        categories = ['Business Credit', 'Digital Payments', 'Active Savings', 'Mobile Banking', 'Emergency Funds']
        user_values = [business_credit, digital_payments, active_savings, mobile_banking, emergency_funds]
        
        # Create radar chart
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=user_values,
            theta=categories,
            fill='toself',
            name='Your Profile',
            line_color='#667eea',
            fillcolor='rgba(102,126,234,0.3)'
        ))
        
        # Add regional average for comparison
        regional_avg_values = [regional_avg] * 5  # Simplified - use regional average for all
        fig_radar.add_trace(go.Scatterpolar(
            r=regional_avg_values,
            theta=categories,
            fill='toself',
            name='Regional Average',
            line_color='#FF6B35',
            fillcolor='rgba(255,107,53,0.1)'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickformat='.0%'
                )
            ),
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Personalized recommendations
        st.markdown("### 💡 Personalized Recommendations")
        
        recommendations = get_personalized_recommendations(inclusion_score, user_inputs, region)
        
        for i, rec in enumerate(recommendations):
            priority_colors = {
                'high': '#E74C3C',
                'medium': '#F39C12',
                'low': '#27AE60'
            }
            
            priority_icons = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢'
            }
            
            color = priority_colors.get(rec['priority'], '#667eea')
            icon = priority_icons.get(rec['priority'], '📌')
            
            st.markdown(f"""
            <div class="recommendation" style="border-left-color: {color};">
                <h4 style="margin-top: 0; color: {color};">{icon} {rec['title']}</h4>
                <p style="margin-bottom: 0;">{rec['description']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional insights based on score
        st.markdown("### 🎯 Next Steps")
        
        if inclusion_score >= 0.75:
            st.success("""
            **🌟 Excellent Financial Health!** You're in the top tier globally. Consider:
            - Exploring investment opportunities
            - Mentoring others in financial inclusion
            - Advocating for financial services in your community
            """)
        elif inclusion_score >= 0.60:
            st.info("""
            **📈 Good Progress!** You're above average but can improve:
            - Focus on your lowest-scoring areas from the radar chart
            - Explore advanced financial products
            - Build stronger emergency fund reserves
            """)
        elif inclusion_score >= 0.40:
            st.warning("""
            **⚡ Action Needed!** Key priorities:
            - Start with mobile banking if available in your region
            - Begin small but regular savings habits
            - Explore microfinance or community lending options
            """)
        else:
            st.error("""
            **🚨 Urgent Support Needed!** Immediate steps:
            - Contact local financial inclusion programs
            - Start with basic mobile money services
            - Seek community financial literacy support
            - Focus on one service at a time to build confidence
            """)
        
        # Impact potential
        st.markdown("### 🌍 Your Impact Potential")
        
        if inclusion_score < regional_avg:
            potential_gain = regional_avg - inclusion_score
            st.markdown(f"""
            <div class="result-card">
                <h3 style="color: #667eea; margin-top: 0;">💪 Improvement Opportunity</h3>
                <p>By reaching your regional average, you could improve by <strong>{potential_gain:.0%}</strong></p>
                <p>This represents significant potential for economic empowerment and financial security.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            leadership_potential = inclusion_score - regional_avg
            st.markdown(f"""
            <div class="result-card">
                <h3 style="color: #27AE60; margin-top: 0;">🏆 Community Leader</h3>
                <p>You're <strong>{leadership_potential:.0%}</strong> above your regional average!</p>
                <p>Consider sharing your knowledge and helping others access financial services.</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #636e72;">
    <p>🌍 <strong>FinScope Global</strong> - Empowering Financial Inclusion Through Data & AI</p>
    <p style="font-size: 0.9em;">Built with Streamlit • Powered by Machine Learning • Data-Driven Insights</p>
</div>
""", unsafe_allow_html=True)