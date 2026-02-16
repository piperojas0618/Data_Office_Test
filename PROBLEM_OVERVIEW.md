# Term Deposit Marketing Campaign - Classification Problem

## Overview

A bank wants to identify which of its clients will accept a **Certificate of Deposit (CDT)** at a fixed term and which will not. The goal is to optimize marketing campaign investments to capture the highest amount of resources possible.

A **classification model** is required to differentiate between clients with a high probability of accepting and those who will not.

## Objective Metric

- **ROC - AUC**

## Modeling Conditions

- The process must be thoroughly documented, explaining the reasoning behind each decision at every stage of the modeling pipeline.

## Development Conditions

- Perform a solid **Exploratory Data Analysis (EDA)**, telling a story with the data.

---

## Dataset Description

### Financial and Demographic Variables

| Variable | Type | Description |
|---|---|---|
| `Edad` | Numeric | Client's age |
| `Tipo_Trabajo` | Categorical | Type of job (`admin.`, `blue-collar`, `entrepreneur`, `housemaid`, `management`, `retired`, `self-employed`, `services`, `student`, `technician`, `unemployed`, `unknown`) |
| `Estado_Civil` | Categorical | Marital status (`divorced`, `married`, `single`, `unknown`). Note: `divorced` includes widowed |
| `Educacion` | Categorical | Education level (`basic.4y`, `basic.6y`, `basic.9y`, `high.school`, `illiterate`, `professional.course`, `university.degree`, `unknown`) |
| `Incumplimiento` | Categorical | Has credit in default? (`no`, `yes`, `unknown`) |
| `Vivienda` | Categorical | Has housing loan? (`no`, `yes`, `unknown`) |
| `Consumo` | Categorical | Has personal loan? (`no`, `yes`, `unknown`) |

### Campaign Contact Variables

| Variable | Type | Description |
|---|---|---|
| `Contacto` | Categorical | Contact communication type (`cellular`, `telephone`) |
| `Mes` | Categorical | Last contact month of year (`jan`, `feb`, ..., `dec`) |
| `Dia` | Categorical | Last contact day of the week (`mon`, `tue`, `wed`, `thu`, `fri`) |
| `Campana` | Numeric | Number of contacts performed during this campaign for this client (includes last contact) |
| `Dias_Ultima_Cam` | Numeric | Days since client was last contacted from a previous campaign (999 = not previously contacted) |
| `No_Contactos` | Numeric | Number of contacts performed before this campaign for this client |
| `Resultado_Anterior` | Categorical | Outcome of the previous marketing campaign (`failure`, `nonexistent`, `success`) |

### Social and Economic Context Variables

| Variable | Type | Description |
|---|---|---|
| `emp_var_rate` | Numeric | Employment variation rate (quarterly indicator) |
| `cons_price_idx` | Numeric | Consumer price index (monthly indicator) |
| `cons_conf_idx` | Numeric | Consumer confidence index (monthly indicator) |
| `euribor3m` | Numeric | Euribor 3-month rate (daily indicator) |
| `nr_employed` | Numeric | Number of employees (quarterly indicator) |

### Target Variable

| Variable | Type | Description |
|---|---|---|
| `y` | Binary | Has the client subscribed a term deposit? (`1` = yes, `0` = no) |
