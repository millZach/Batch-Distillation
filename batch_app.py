import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.animation as animation
import time
from sympy import *


@st.cache
def load_data(filename):
    # Storing data so it is not constantly reloaded
    data = pd.read_csv(filename)
    return data


@st.cache
def image_display(image_filename):
    image = Image.open(image_filename)
    new_image = image.resize((850, 600))
    return new_image


# A - ethyl acetate, B - butyl acetate
T_bp_A = 77.1  # C
mw_A = 88.1  # g/mol
rho_A = 0.902  # g/mL
T_bp_B = 126  # C
mw_B = 116.16  # g/mol
rho_B = 0.882  # g/mol

properties_dict = {
    'Parameter': ['Boiling Point (K)', 'Molecular Weight (g/mol)', 'Density (g/mL)'],
    'Ethyl Acetate': [T_bp_A, mw_A, rho_A],
    'Butyl Acetate': [T_bp_B, mw_B, rho_B]
}
properties_df = pd.DataFrame(properties_dict)
properties_df.set_index('Parameter', inplace=True)

vle_df = load_data('VLE_data.csv')

# Fitting a line to data from CSV
fit_data = np.polyfit(vle_df['x_A'], vle_df['y_A'], 5)
xline = np.linspace(0, 1, 101)
func = np.poly1d(fit_data)
yfit = np.polyval(fit_data, xline)

Tx_fit_data = np.polyfit(vle_df['x_A2'], vle_df['Temperature 2'], 5)
Ty_fit_data = np.polyfit(vle_df['y_A2'], vle_df['Temperature 2'], 5)
Tx_func = np.poly1d(Tx_fit_data)
Ty_func = np.poly1d(Ty_fit_data)
Tx_fit = np.polyval(Tx_fit_data, xline)
Ty_fit = np.polyval(Ty_fit_data, xline)


st.title("Batch Distillation of Ethyl and Butyl Acetate")
page_select = st.sidebar.selectbox('Pages',
                                   ('General Information', 'Theoretical Simulation', 'Wet Lab 1', 'Wet Lab 2'))

# Displays the general Information page if selected
if page_select == 'General Information':
    st.markdown(
        '**Table 1** shows the parameters used for this analysis. Values were found using https://webbook.nist.gov/.')
    st.table(properties_df)
    st.markdown(
        'This is a process flow diagram of the apparatus used for this distillation:')
    st.image(image_display('Batch Distillation.JPG'))

    st.markdown(
        'Vapor liquid equilibrium data for Ethyl and Butyl acetate at 1 atm. Data was provided by http://vle-calc.com/phase_diagram.html')

    # Straight line
    x_straight = np.linspace(0, 1, 10)
    y_straight = x_straight

    # Liquid-vapor composition
    x_A1 = np.array(vle_df['x_A'])
    y_A1 = np.array(vle_df['y_A'])

    # Bubble/dewpoitn graph
    x_A2 = np.array(vle_df['x_A2'])
    y_A2 = np.array(vle_df['y_A2'])
    T_A2 = np.array(vle_df['Temperature 2'])

    # Plotting the vle data
    fig = plt.subplots(figsize=(14, 7), dpi=300)

    plt.subplot(1, 2, 1)
    plot_1 = plt.plot(
        x_A1, y_A1, '-b',
        x_straight, y_straight, '-k',
        lw=3
    )
    plt.title('Liquid-Vapor Composition', fontsize=20)
    plt.xlabel(r'$x_{EtAc}$', fontsize=20)
    plt.ylabel(r'$y_{EtAc}$', fontsize=20)

    plt.subplot(1, 2, 2)
    plot_2 = plt.plot(
        x_A2, T_A2, '-b',
        y_A2, T_A2, '-r',
        lw=3
    )
    plt.title('Bubble/Dewpoint (T-x-y)', fontsize=20)
    plt.xlabel(r'$x-y_{EtAc}$', fontsize=20)
    plt.ylabel(r'$Temperature (^\circ C)$', fontsize=16)
    plt.legend(['Bubble Point', 'Dew Point'])

    plt.tight_layout()
    # Displays plots in Stream lit
    st.pyplot()

    # Check box to show data
    show_vle_data = st.checkbox('Show All Data')
    if show_vle_data:
        st.table(vle_df)


if page_select == 'Theoretical Simulation':

    q_b = st.sidebar.number_input(
        'Boiler Heat (W)', min_value=0.0, value=140.0)
    x_AF = st.sidebar.slider('x_EtAc_F', 0.01, 0.99, 0.5)
    x_AW_final = st.sidebar.slider('x_EtAc_final', 0.01, x_AF, 0.01)
    y_AF_disp = st.sidebar.text(f'y_EtAc_F = {round(func(x_AF),2)}')
    y_AF = func(x_AF)
    F = (x_AF * 500) * rho_A / mw_A + \
        ((1 - x_AF) * 500) * rho_B / mw_B
    F_disp = st.sidebar.text(f"F = {round(F,2)} mole")
    h_vap_A = st.sidebar.number_input(
        'Heat of Vaporiztion EtAc (J/mol)', min_value=0.0, value=35000.0)
    h_vap_B = st.sidebar.number_input(
        'Heat of Vaporiztion BtAc (J/mol)', min_value=0.0, value=43000.0)
    st.sidebar.markdown('**Volume basis is 500 mL**')

    st.markdown('''
        Based on starting parameters listed in the side bar, the plots below show the vapor-liquid equilibrium curve
        and the bubble/dewpoint temeperature curve for the mixture. The equilibrium temperature is the boiling point current
        boiling point of the mixture.
        ''')
    # Straight line
    x_straight = np.linspace(0, 1, 10)
    y_straight = x_straight

    # Plotting the vle data
    fig = plt.subplots(figsize=(14, 7), dpi=300)

    plt.subplot(1, 2, 1)
    plot_1 = plt.plot(
        xline, yfit, '-b',
        x_straight, y_straight, '-k',
        x_AF, func(x_AF), 'ok',
        ms=10, lw=3
    )
    plt.title('Liquid-Vapor Composition', fontsize=20)
    plt.xlabel(r'$x_{EtAc}$', fontsize=20)
    plt.ylabel(r'$y_{EtAc}$', fontsize=20)

    plt.subplot(1, 2, 2)
    plot_2 = plt.plot(
        xline, Tx_fit, '-b',
        xline, Ty_fit, '-r',
        x_AF, Tx_func(x_AF), 'ok',
        y_AF, Ty_func(y_AF), 'ok',
        ms=10, lw=3
    )
    plt.title('Bubble/Dewpoint (T-x-y)', fontsize=20)
    plt.xlabel(r'$x-y_{EtAc}$', fontsize=20)
    plt.ylabel(r'$Temperature (^\circ C)$', fontsize=16)
    plt.legend(['Bubble Point', 'Dew Point'])
    plt.text(0, 80, f"Equilibrium Temperature: {round(Tx_func(x_AF),2)}" + r'$^\circ C$', fontsize=16)

    plt.tight_layout()
    # Displays plots in Stream lit
    st.pyplot()

    # Setting up simulation animation lists
    index_start = int(np.where(xline == x_AW_final)[0])
    index_finish = int(np.where(xline == x_AF)[0])
    x_AW = xline[index_start:index_finish + 1]
    x_AW = x_AW[::-1]
    temp_lst = Tx_fit[index_start:index_finish + 1]
    temp_lst = temp_lst[::-1]
    P_sat_A = []
    P_sat_B = []
    rel_vol = []
    W = []
    D = []
    D_dot = []
    t = []
    x_AD_avg = []

    run_sim = st.button('Run Simulation')
    if run_sim:
        progress_bar = st.sidebar.progress(0)
        frame_text = st.sidebar.empty()

        for i in range(0, len(x_AW)):
            # Saturation pressure of ethyl and butyl acetate, bar
            P_sat_A.append(np.exp(9.5314 - (2790.5 / (temp_lst[i] + 216))))
            P_sat_B.append(np.exp(9.5634 - (3151.09 / (temp_lst[i] + 204))))
            # Relative volitility of the mixture
            rel_vol.append(P_sat_A[i] / P_sat_B[i])
            # Amount of moles of the mixture left in the bottoms, mole
            W.append(F / (np.exp(1 / (rel_vol[i] - 1) * np.log((x_AF / x_AW[i]) * (
                (1 - x_AW[i]) / (1 - x_AF))) + np.log((1 - x_AW[i]) / (1 - x_AF)))))
            # Amount of distillate produced, mole
            D.append(F - W[i])
            # Rate at which distillate is being transfered assumed 25% is transferred, mole/s
            D_dot.append((
                q_b / ((h_vap_A * x_AW[i] + h_vap_B * (1 - x_AW[i])) / 2)) * 0.20)
            # time the experiment has been running, mins
            t.append(D[i] / (D_dot[i] * 60))
            if D[i] > 0:
                x_AD_avg.append((x_AF * F - x_AW[i] * W[i]) / D[i])
            else:
                x_AD_avg.append(0)

            progression = int((i / (len(x_AW) - 1)) * 100)
            progress_bar.progress(progression)
            frame_text.text(f"Distillation Time: {round(t[i],0)} minutes")
            time.sleep(0.1)

        # Plotting the vle data
        fig_sim = plt.figure()

        plot_1 = plt.plot(
            xline, yfit, '-b',
            x_straight, y_straight, '-k',
            x_AW[1:-1], func(x_AW[1:-1]), 'xk',
            x_AW[0], func(x_AW[0]), 'og',
            x_AW[-1], func(x_AW[-1]), 'or',
            ms=10, lw=3
        )
        plt.title('Liquid-Vapor Composition', fontsize=20)
        plt.xlabel(r'$x_{EtAc}$', fontsize=20)
        plt.ylabel(r'$y_{EtAc}$', fontsize=20)
        plt.legend(['VLE data', 'Raoults Law', 'Theoretical Data',
                    'Start Distillation', 'Stop Distillation'], loc=4)
        plt.tight_layout()
        st.pyplot(fig_sim)

        fig_sim2, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        plt_1 = ax1.plot(
            t, D, '-r',
            t, W, '-b',
            lw=2
        )
        plt_2 = ax2.plot(t[1:], x_AD_avg[1:], 'xg', ms=10)
        ax1.set_xlabel('Time (minutes)', fontsize=16)
        ax1.set_ylabel('Bottoms/Distillate (mole)', fontsize=16)
        ax2.set_ylabel(r'$x_{AD,avg}$', fontsize=16)
        plt.title('Theoretical Data', fontsize=20)
        leg = plt_1 + plt_2
        ax1.legend(leg, ['Distillate', 'Bottoms', r'$x_{AD,avg}$'], loc=6)

        plt.tight_layout()
        st.pyplot(fig_sim2)
        vle_dict = {
            'x_AW': x_AW,
            'W (gmole)': W,
            'D (gmole)': D,
            'x_AD_avg': x_AD_avg,
            'Time (min)': t,
            'Saturated Pressure A (bar)': P_sat_A,
            'Saturated Pressure B (bar)': P_sat_A,
            'Relative volatility': rel_vol,
            'Temperature (C)': temp_lst
        }

        theoretical_df = pd.DataFrame(vle_dict)
        st.table(theoretical_df)

    show_eqns = st.sidebar.checkbox('Show Equations Used')
    if show_eqns:
        st.markdown(
            'In this section I will walk through equations that were used to generate theoretical data.')
        st.markdown(
            'The following equation is used to calculate how many moles the bottoms flask is charged with:')
        st.latex(r'''
            F = \frac{V_{EtAc}\cdot \rho_{EtAc}}{MW_{EtAc}} + \frac{V_{BtAc}\cdot \rho_{BtAc}}{MW_{BtAc}}
            ''')

        st.latex(r'''
            F = \frac{250 mL\cdot 0.902\frac{g}{mL}}{88.1\frac{g}{mole}} + \frac{250 mL\cdot 0.882\frac{g}{mL}}{116.16\frac{g}{mole}}
            = 4.46 mole
            ''')


if page_select == 'Wet Lab 1':
    st.markdown('''
        Data collected during wet lab 1 from a Vernier Mini GC. The ratios of ethyl/butyl acetate were used:
        **Run 1: 0.05/0.95, Run 2: 0.95/0.05, Run 3: 0.50/0.50, Run 4: 0.75/0.25, Run 5: 0.25/0.75**
        ''')

    GC_data = load_data('CHE415_Team25_WT1GC.csv')

    fig_gc = plt.figure()

    plt.plot(
        GC_data['Run 1: time (min)'], GC_data['Run 1: signal (mV)'], '-k',
        GC_data['Run 2: time (min)'], GC_data['Run 2: signal (mV)'], '-b',
        GC_data['Run 3: time (min)'], GC_data['Run 3: signal (mV)'], '-g',
        GC_data['Run 4: time (min)'], GC_data['Run 4: signal (mV)'], '-r',
        GC_data['Run 5: time (min)'], GC_data['Run 5: signal (mV)'], '-y',
    )
    plt.xlabel('Time (min)')
    plt.ylabel('Signal (mV)')
    plt.legend(['Run 1', 'Run 2', 'Run 3', 'Run 4', 'Run 5'], loc=0)
    plt.ylim((0, 1600))
    txt_1 = plt.text(5.5, 1300, 'Butyl Acetate')
    txt_2 = plt.text(1.3, 1450, 'Ethyl Acetate')

    st.pyplot()

    st.markdown('''
        Once the data was collected a calibration curve was generated for the area under each respective
        peak vs the mass fraction of each substance, shown below.
        ''')

    x_etoac = [0.05, 0.25, 0.5, 0.75, 0.95]
    y_etoac = [1.76, 10.77, 23.19, 43.7, 82.11]

    x_btoac = [0.05, 0.25, 0.5, 0.75, 0.95]
    y_btoac = [17.89, 56.5, 76.81, 89.23, 98.24]

    calibration_dict = {
        'Mass Fraction EtOAc': [0.05, 0.25, 0.5, 0.75, 0.95],
        'Pct Area EtOAc': [1.76, 10.77, 23.19, 43.7, 82.11],
        'Mass Fraction BtOAc': [0.05, 0.25, 0.5, 0.75, 0.95],
        'Pct Area BtOAc': [17.89, 56.5, 76.81, 89.23, 98.24],
    }

    calibration_df = pd.DataFrame(calibration_dict)

    fit_cal_etoac = np.polyfit(x_etoac, y_etoac, 4)
    xline = np.linspace(0.05, 0.95, 101)
    f_cal_etoac = np.poly1d(fit_cal_etoac)
    yfit_etoac = np.polyval(fit_cal_etoac, xline)

    fit_cal_btoac = np.polyfit(x_btoac, y_btoac, 4)
    f_cal_btoac = np.poly1d(fit_cal_btoac)
    yfit_btoac = np.polyval(fit_cal_btoac, xline)

    pct_area_etoac = st.sidebar.number_input(
        'Area (%) EtOAc', min_value=1.76, max_value=82.11)
    pct_area_btoac = st.sidebar.number_input(
        'Area (%) BtOAc', min_value=17.89, max_value=98.24)

    c_1_et = fit_cal_etoac[0]
    c_2_et = fit_cal_etoac[1]
    c_3_et = fit_cal_etoac[2]
    c_4_et = fit_cal_etoac[3]
    c_5_et = fit_cal_etoac[4]

    c_1_bt = fit_cal_btoac[0]
    c_2_bt = fit_cal_btoac[1]
    c_3_bt = fit_cal_btoac[2]
    c_4_bt = fit_cal_btoac[3]
    c_5_bt = fit_cal_btoac[4]

    x = Symbol('x')

    eqn_etoac = Eq(c_1_et * x**4 + c_2_et * x**3 + c_3_et *
                   x**2 + c_4_et * x + c_5_et, pct_area_etoac)
    eqn_btoac = Eq(c_1_bt * x**4 + c_2_bt * x**3 + c_3_bt *
                   x**2 + c_4_bt * x + c_5_bt, pct_area_btoac)

    m_frac_etoac = round(solve(eqn_etoac)[1], 3)
    m_frac_btoac = round(solve(eqn_btoac)[0], 3)

    st.sidebar.text(f'Mass Fraction EtOAc: {m_frac_etoac}')
    st.sidebar.text(f'Mass Fraction BtOAc: {m_frac_btoac}')

    fig_cal = plt.figure()

    plt.plot(
        x_etoac, y_etoac, 'ok',
        x_btoac, y_btoac, 'xk',
        xline, yfit_etoac, '-g',
        xline, yfit_btoac, '-y',
        m_frac_etoac, pct_area_etoac, 'ob',
        m_frac_btoac, pct_area_btoac, 'or',
        ms=10
    )

    plt.legend(['EtOAc', 'BtOAc', 'EtOAc Fit', 'BtOAc Fit'])
    plt.xlabel('Mass Fraction (EtOAc/BtOAc)')
    plt.ylabel('Area (%)')

    st.pyplot()

    st.table(calibration_df)
