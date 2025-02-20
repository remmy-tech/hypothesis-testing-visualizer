import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t, f

# Title
st.title("Hypothesis Testing Visualizer")

# Sidebar Inputs
st.sidebar.header("Settings")

distribution = st.sidebar.selectbox("Select Distribution:", ["Normal", "t-distribution", "F-distribution"])
alpha = st.sidebar.slider("Significance Level (α):", min_value=0.001, max_value=0.1, value=0.05, step=0.001)
test_type = st.sidebar.radio("Test Type:", ["Two-sided", "One-sided (right)", "One-sided (left)"])
statistic = st.sidebar.number_input("Test Statistic:", value=0.0)

# Degrees of freedom sliders (only when needed)
df1 = None
df2 = None
if distribution in ["t-distribution", "F-distribution"]:
    df1 = st.sidebar.slider("Degrees of Freedom (df1):", min_value=1, max_value=30, value=5, step=1)
if distribution == "F-distribution":
    df2 = st.sidebar.slider("Degrees of Freedom (df2):", min_value=1, max_value=500, value=100, step=1)

# Function to plot the hypothesis test
def plot_hypothesis_test(distribution, alpha, test_type, statistic, df1, df2):
    mu, sigma = 0, 1  # mean and standard deviation for normal and t-distribution
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

    if distribution == 'Normal':
        y = norm.pdf(x, mu, sigma)
        dist = norm
    elif distribution == 't-distribution':
        y = t.pdf(x, df1)
        dist = t(df1)
    else:  # F-distribution
        x = np.linspace(0, 5, 100)
        y = f.pdf(x, df1, df2)
        dist = f(df1, df2)

    # Critical values
    if distribution in ['Normal', 't-distribution']:
        if test_type == 'Two-sided':
            critical_pos = dist.ppf(1 - alpha/2)
            critical_neg = -critical_pos
        elif test_type == 'One-sided (right)':
            critical_pos = dist.ppf(1 - alpha)
            critical_neg = None
        elif test_type == 'One-sided (left)':
            critical_pos = None
            critical_neg = dist.ppf(alpha)
    else:  # F-distribution
        critical_pos = dist.ppf(1 - alpha)
        critical_neg = None

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the distribution
    ax.plot(x, y, 'b-', lw=2, label=f'{distribution} distribution')

    # Fill acceptance and rejection regions
    if test_type == 'Two-sided' and distribution != 'F-distribution':
        ax.fill_between(x, y, where=(x >= critical_neg) & (x <= critical_pos), alpha=0.2, color='lightblue')
    elif test_type == 'One-sided (right)' or (distribution == 'F-distribution' and test_type == 'Two-sided'):
        ax.fill_between(x, y, where=(x <= critical_pos), alpha=0.2, color='lightblue')
    elif test_type == 'One-sided (left)' and distribution != 'F-distribution':
        ax.fill_between(x, y, where=(x >= critical_neg), alpha=0.2, color='lightblue')

    # Fill rejection regions
    if critical_neg is not None:
        ax.fill_between(x, y, where=(x < critical_neg), alpha=0.2, color='blue')
    if critical_pos is not None:
        ax.fill_between(x, y, where=(x > critical_pos), alpha=0.2, color='blue')

    # Adding critical values
    if critical_pos is not None:
        ax.axvline(x=critical_pos, color='blue', linestyle='--', label='Critical Value')
    if critical_neg is not None and distribution != 'F-distribution':
        ax.axvline(x=critical_neg, color='blue', linestyle='--')

    # Adding test statistic
    ax.axvline(x=statistic, color='red', linestyle='--', label='Test Statistic')

    # Calculate the p-value
    if test_type == 'Two-sided' and distribution != 'F-distribution':
        p_value = 2 * (1 - dist.cdf(abs(statistic)))
    elif test_type == 'One-sided (right)' or (distribution == 'F-distribution' and test_type == 'Two-sided'):
        p_value = 1 - dist.cdf(statistic)
    elif test_type == 'One-sided (left)' and distribution != 'F-distribution':
        p_value = dist.cdf(statistic)

    # Title and labels
    ax.set_title(f'{distribution} Distribution Curve with Hypothesis Testing')
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability Density')

    # Grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    st.pyplot(fig)

    # Display results
    st.write(f"### Results")
    st.write(f"- **Critical Value(s):** {critical_neg:.3f}, {critical_pos:.3f}")
    st.write(f"- **Test Statistic:** {statistic:.3f}")
    st.write(f"- **P-value:** {p_value:.4f}")
    if (statistic > critical_pos if critical_pos is not None else False) or (statistic < critical_neg if critical_neg is not None else False):
        st.write("### 🚨 Reject the null hypothesis.")
    else:
        st.write("### ✅ Fail to reject the null hypothesis.")

# Call function
plot_hypothesis_test(distribution, alpha, test_type, statistic, df1, df2)