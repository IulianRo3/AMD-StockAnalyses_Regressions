from pprint import pprint
from sys import stdout
import sympy as sym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
#                ROMANA               //            ENGLISH
#1.Definirea si apelarea unei functii // Defining and calling the functions
#2.Utilizarea pachetului scikit-learn // Using the package of scikit-learn
def optimise_pls_cv(X, y, n_comp, plot_components=True):
    #Rulam PLS incluzand un numar variabil de componente pana la n si calculam SSE // Running the PLS including a variable number of components up to n and we calculate SSE
    sse = []
    component = np.arange(1, n_comp)
    for i in component:
        pls = PLSRegression(n_components=i)
        # Cross-validare // Cross-Validation
        y_cv = cross_val_predict(pls, X, y, cv=10)
        sse.append(mean_squared_error(y, y_cv))
        comp = 100 * (i + 1) / 40
        # Procedeu pentru actualizarea statutului pe aceeasi linie // Process for updating the status on the same line of code
        stdout.write("\r%d%% completat" % comp)
        stdout.flush()
    stdout.write("\n")
    # Calculeaza si afiseaza pozitia minimului SSE // Calculate and print the minimum position of SSE
    ssemin = np.argmin(sse)
    print("Numar de componente sugerate: ", ssemin + 1)
    stdout.write("\n")
    if plot_components is True:
        with plt.style.context(('ggplot')):
            plt.plot(component, np.array(sse), '-v', color='blue', mfc='blue')
            plt.plot(component[ssemin], np.array(sse)[ssemin], 'P', ms=10, mfc='red')
            plt.xlabel('Numar de componente pentru Partial Least Squares')
            plt.ylabel('SSE')
            plt.title('PLS')
            plt.xlim(left=-1)
        plt.show()
    # Defineste PLS-ul cu un numar optim de componente // Defines the PLS with an optimal number of components
    pls_opt = PLSRegression(n_components=ssemin + 1)
    # Fit pentru intreg setul de date // Fitting the entire dataset
    pls_opt.fit(X, y)
    y_c = pls_opt.predict(X)
    # Cross-validare // Cross-validation
    y_cv = cross_val_predict(pls_opt, X, y, cv=10)
    # Calculeaza valori pentru calibrare si cross-validare // Calculate valors for calibration and cross-validation
    score_c = r2_score(y, y_c)
    score_cv = r2_score(y, y_cv)
    # Calculeaza SSE pentru calibrare si cross validare // Calculate SSE for calibration and cross-validation
    sse_c = mean_squared_error(y, y_c)
    sse_cv = mean_squared_error(y, y_cv)
    print('R2 calib: %5.3f' % score_c)
    print('R2 CV: %5.3f' % score_cv)
    print('SSE calib: %5.3f' % sse_c)
    print('SSE CV: %5.3f' % sse_cv)
    # Plot cu regresie si SSE // Plot with regression and SSE
    rangey = max(y) - min(y)
    rangex = max(y_c) - min(y_c)
    # Proiecteaza o linie intre cross validare si SSE // Draws a line between cross-validation and SSE
    z = np.polyfit(y, y_c, 1)
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(y_c, y, c='red', edgecolors='k')
        # Plot the best fit line 
        ax.plot(np.polyval(z, y), y, c='blue', linewidth=1)
        # Plot the ideal 1:1 line
        ax.plot(y, y, color='green', linewidth=1)
        plt.title('$R^{2}$ (CV): ' + str(score_cv))
        plt.xlabel('PREVIZIONAT')
        plt.ylabel('MASURAT')
        plt.show()
    return


#3.Import de fisier CSV in pandas // Importing the CVS files in pandas
AMD = pd.read_csv('AMD.csv')
print('------------1----------------')
pprint(AMD)
#4.Accesarea datelor cu loc si iloc // Accesing data with loc and iloc
print('---------2---------')
pprint(AMD.iloc[[0, 1, 2], [0, 1, 2]])
pprint(AMD.loc[1:10, ['Open']])

AMD2 = AMD.loc[0:22,['Date','Open','Close']]
print(AMD2)
#5.Tratarea valorilor lipsa // Fixing with the missing values
print('----------3-----------')
print('Exista valori NULL in CSV-ul nostru si daca da, cate: ', AMD.isnull().values.any().sum(), '\n')
#6.Utilizarea pachetelor statmodels // Using the statsmodel packages
print('----------4-----------')
target = pd.DataFrame(AMD, columns=['High'])
print(AMD.dtypes)
X = AMD[['Open', "Close", 'Low']]
Y = target['High']
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
predictions = model.predict(X)
pprint(model.summary())

AMD['Date'] = AMD['Date'].astype('datetime64[ns]')
AMD['Date'].dt.round('30d')
pprint(AMD['Date'].tail(15))
#7.Reprezentarea grafica a datelor // Graphic representation of dataset
optimise_pls_cv(X, Y, 4, plot_components=True)
X = np.array(AMD['Date'], dtype=float)
Y = np.array(AMD['Open'], dtype=float)
plt.plot(X, Y, 'ro',label="Original Data")
def func(x, a, b, c, d):
    return a *x**3+ b*x**2+c*x+d
popt, pcov = curve_fit(func, X, Y)
print("a = %s , b = %s, c = %s, d = %s" % (popt[0], popt[1], popt[2], popt[3]))
xs = sym.Symbol('\lambda')
tex = sym.latex(func(xs,*popt)).replace('$', '')
plt.title(r'$f(\lambda)= %s$' %(tex),fontsize=16)
plt.plot(X, func(X, *popt), label="Fitted Curve") 
plt.legend(loc='upper left')

plt.show()


X = np.array(AMD['Date'], dtype=float)
Y = np.array(AMD['Open'], dtype=float)
plt.plot(X, Y, 'ro',label="Original Data")
def func(x, a, b, c, d,e,f,g):
    #return a*x*6 + b*x5 +c*x4 + d*x3 +e*x*2 + f*x + g  Trying out a polynomial fit
    return a *x+ b
popt, pcov = curve_fit(func, X, Y)
print("a = %s , b = %s, c = %s, d = %s" % (popt[0], popt[1], popt[2], popt[3]))
xs = sym.Symbol('\lambda')
tex = sym.latex(func(xs,*popt)).replace('$', '')
plt.title(r'$f(\lambda)= %s$' %(tex),fontsize=16)
plt.plot(X, func(X, *popt), label="Fitted Curve") 
plt.legend(loc='upper left')
plt.show()

#8. Stergerea de coloane si inregistrari // Deleting of colums and records 
AMD.drop(['Volume'],axis=1,inplace=True) #Stergerea unei coloane deoarece nu ne folosea // Deleting a column because it was redundant
AMD.drop([0,1],axis=0,inplace=True) #Stergerea a 2 observatii pentru a incepe de la inceputul lunii // Deleting 2 observations to start from the beginning
#9
AMD=AMD.round(3) # Rotunjirea la 3 zecimale // Rounding to 3 decimal
print('------------aaa----------------')
print(AMD)
#10.Modificarea datelor in pachetul pandas // Data modification in the pandas package
print(AMD.describe().round(3).iloc[0:22 , 0:2]) #Observatiile pentru luna Mai folosind iloc // May observations using iloc
AMD.loc[5,['Open']]=27.002
print('------------AAA----------------')
print(AMD)