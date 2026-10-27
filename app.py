"""
Real Estate Investment Advisor - Streamlit app (single file)
Run order:  python eda.py   (creates data/cleaned_housing.csv)   then   streamlit run app.py
Needs: streamlit, pandas, numpy
"""
import os
import numpy as np, pandas as pd, streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
CURRENT_YEAR, AMENITIES = 2025, ["Gym", "Pool", "Clubhouse", "Garden", "Playground"]

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")


@st.cache_data
def load():
    p = os.path.join(BASE, "data", "cleaned_housing.csv")
    if not os.path.exists(p):
        st.error("data/cleaned_housing.csv not found. Run `python eda.py` first."); st.stop()
    return pd.read_csv(p)


df = load()
city_median = df.groupby("City")["Price_per_SqFt_INR"].median()


def analyse(i):
    """Same rules used in eda.py: growth assumption + good-investment rule."""
    transport = {"Low": 0, "Medium": 1, "High": 2}[i["transport"]]
    infra = (i["schools"] / 10 + i["hospitals"] / 10 + transport / 2) / 3
    age = CURRENT_YEAR - i["year"]
    ready = int(i["avail"] == "Ready_to_Move")
    growth = float(np.clip(0.06 + 0.02 * transport / 2 + 0.02 * infra + 0.01 * len(i["amen"]) / 5 - 0.0004 * age + 0.005 * ready, 0.03, 0.12))
    ppsf = i["price"] * 1e5 / i["size"]; med = float(city_median[i["city"]])
    quality = int(i["bhk"] >= 3) + ready + int(transport == 2) + int(i["parking"]) + int(i["security"]) + int(infra >= 0.5)
    return dict(growth=growth, future=i["price"] * (1 + growth) ** 5, ppsf=ppsf, med=med, quality=quality,
                good=(ppsf <= med) and quality >= 3)


st.title("🏠 Real Estate Investment Advisor")
st.caption("Verdict = price per sq ft at or below the city median AND quality score ≥ 3 of 6. "
           "5-year price uses an assumed growth rate (the dataset has no price history).")
t1, t2, t3 = st.tabs(["🔮 Predict", "🔎 Explore listings", "📊 Market insights"])

with t1:
    a, b, c = st.columns(3)
    with a:
        state = st.selectbox("State", sorted(df.State.unique()))
        city = st.selectbox("City", sorted(df[df.State == state].City.unique()))
        ptype = st.selectbox("Property type", sorted(df.Property_Type.unique()))
        bhk = st.slider("BHK", 1, 5, 3)
        size = st.number_input("Size (sq ft)", 400, 6000, 2500, 50)
        price = st.number_input("Asking price (₹ Lakhs)", 10.0, 1000.0, 250.0, 5.0)
    with b:
        year = st.slider("Year built", 1990, 2025, 2012)
        avail = st.selectbox("Availability", sorted(df.Availability_Status.unique()))
        transport = st.select_slider("Public transport", ["Low", "Medium", "High"], "Medium")
        parking = st.checkbox("Parking", True); security = st.checkbox("Security", True)
    with c:
        schools = st.slider("Nearby schools", 0, 10, 5); hospitals = st.slider("Nearby hospitals", 0, 10, 5)
        amen = st.multiselect("Amenities", AMENITIES, ["Gym", "Garden"])
    if st.button("Analyse property", type="primary"):
        r = analyse(dict(city=city, bhk=bhk, size=size, price=price, year=year, avail=avail, transport=transport,
                         parking=parking, security=security, schools=schools, hospitals=hospitals, amen=amen))
        x, y, z = st.columns(3)
        x.metric("Verdict", "✅ Good investment" if r["good"] else "⚠️ Not recommended")
        y.metric("Estimated price in 5 years", f"₹{r['future']:,.0f} L", f"{r['future'] / price - 1:+.0%}")
        z.metric("Assumed annual growth", f"{r['growth']:.1%}")
        st.write(f"Price per sq ft **₹{r['ppsf']:,.0f}** vs {city} median **₹{r['med']:,.0f}** · quality score **{r['quality']}/6**")

with t2:
    f1, f2, f3 = st.columns(3)
    states = f1.multiselect("State", sorted(df.State.unique())); types = f2.multiselect("Type", sorted(df.Property_Type.unique()))
    bhks = f3.multiselect("BHK", [1, 2, 3, 4, 5])
    pr = st.slider("Price (₹ Lakhs)", 10, 500, (10, 500)); good = st.checkbox("Only good investments")
    m = df[df.Price_in_Lakhs.between(*pr)]
    if states: m = m[m.State.isin(states)]
    if types: m = m[m.Property_Type.isin(types)]
    if bhks: m = m[m.BHK.isin(bhks)]
    if good: m = m[m.Good_Investment == 1]
    st.write(f"**{len(m):,}** properties")
    st.dataframe(m[["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Future_Price_5y", "Good_Investment"]].head(500), use_container_width=True)

with t3:
    l, r_ = st.columns(2)
    l.subheader("Avg price per sq ft by state (₹)"); l.bar_chart(df.groupby("State").Price_per_SqFt_INR.mean().sort_values())
    r_.subheader("Good-investment rate, top 15 cities"); r_.bar_chart(df.groupby("City").Good_Investment.mean().sort_values(ascending=False).head(15))
    l.subheader("Avg price by property type (₹ L)"); l.bar_chart(df.groupby("Property_Type").Price_in_Lakhs.mean())
    r_.subheader("Listings by BHK"); r_.bar_chart(df.BHK.value_counts().sort_index())
    st.caption("Differences between cities and states are small in this dataset, so read the charts as relative, not as strong market signals.")
