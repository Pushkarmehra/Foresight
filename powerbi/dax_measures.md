# 📐 Essential DAX Measures for Foresight Fleet Analytics

Copy and paste these measures directly into Power BI:

---

### 1. Fleet Capacity & Inventory
```dax
Total Engines = 
DISTINCTCOUNT('train_FD001_processed'[unit_nr])
```

```dax
Total Operational Cycles = 
COUNTROWS('train_FD001_processed')
```

---

### 2. Fleet Health & Predictive Risk KPIs

#### Critical Engines Count (Imminent Failure within 30 Cycles)
```dax
Critical Engines Count = 
CALCULATE(
    DISTINCTCOUNT('train_FD001_processed'[unit_nr]),
    FILTER(
        'train_FD001_processed',
        'train_FD001_processed'[failure_in_30_cycles] = 1 &&
        'train_FD001_processed'[time_cycles] = CALCULATE(MAX('train_FD001_processed'[time_cycles]), ALLEXCEPT('train_FD001_processed', 'train_FD001_processed'[unit_nr]))
    )
)
```

#### Average Fleet RUL
```dax
Average Current RUL = 
AVERAGEX(
    VALUES('train_FD001_processed'[unit_nr]),
    CALCULATE(
        MIN('train_FD001_processed'[rul_clipped]),
        'train_FD001_processed'[time_cycles] = CALCULATE(MAX('train_FD001_processed'[time_cycles]), ALLEXCEPT('train_FD001_processed', 'train_FD001_processed'[unit_nr]))
    )
)
```

#### Dynamic Engine Status Indicator
```dax
Engine Risk Status = 
VAR CurrentRUL = SELECTEDVALUE('train_FD001_processed'[rul_clipped])
RETURN
    SWITCH(
        TRUE(),
        CurrentRUL <= 30, "Critical (<= 30 cycles)",
        CurrentRUL <= 60, "Warning (31 - 60 cycles)",
        "Healthy (> 60 cycles)"
    )
```

#### Health Status Color Code (Hex Format for Conditional Formatting)
```dax
Health Status Color = 
VAR CurrentRUL = SELECTEDVALUE('train_FD001_processed'[rul_clipped])
RETURN
    SWITCH(
        TRUE(),
        CurrentRUL <= 30, "#FF3B30",   -- Red (Critical)
        CurrentRUL <= 60, "#FF9500",   -- Amber (Warning)
        "#34C759"                      -- Green (Healthy)
    )
```
