"""
FastAPI приложение для Credit Default Prediction API
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os
from pathlib import Path


# Определение схемы входных данных с помощью Pydantic
class ClientData(BaseModel):
    """Схема данных клиента для предсказания"""
    LIMIT_BAL: float = Field(..., description="Лимит кредитной карты", example=20000.0)
    SEX: int = Field(..., description="Пол (1=мужчина, 2=женщина)", example=2, ge=1, le=2)
    EDUCATION: int = Field(..., description="Образование (1-4)", example=2, ge=1, le=4)
    MARRIAGE: int = Field(..., description="Семейное положение (1-3)", example=1, ge=1, le=3)
    AGE: int = Field(..., description="Возраст", example=24, ge=18, le=100)
    PAY_0: int = Field(..., description="Статус платежа в сентябре", example=2, ge=-2, le=8)
    BILL_AMT1: float = Field(..., description="Сумма счета в сентябре", example=3913.0)
    PAY_AMT1: float = Field(..., description="Сумма платежа в сентябре", example=0.0)
    
    # Опциональные поля (будут заполнены значениями по умолчанию если не переданы)
    PAY_2: int = Field(0, description="Статус платежа в августе", ge=-2, le=8)
    PAY_3: int = Field(0, description="Статус платежа в июле", ge=-2, le=8)
    PAY_4: int = Field(0, description="Статус платежа в июне", ge=-2, le=8)
    PAY_5: int = Field(0, description="Статус платежа в мае", ge=-2, le=8)
    PAY_6: int = Field(0, description="Статус платежа в апреле", ge=-2, le=8)
    
    BILL_AMT2: float = Field(0.0, description="Сумма счета в августе")
    BILL_AMT3: float = Field(0.0, description="Сумма счета в июле")
    BILL_AMT4: float = Field(0.0, description="Сумма счета в июне")
    BILL_AMT5: float = Field(0.0, description="Сумма счета в мае")
    BILL_AMT6: float = Field(0.0, description="Сумма счета в апреле")
    
    PAY_AMT2: float = Field(0.0, description="Сумма платежа в августе")
    PAY_AMT3: float = Field(0.0, description="Сумма платежа в июле")
    PAY_AMT4: float = Field(0.0, description="Сумма платежа в июне")
    PAY_AMT5: float = Field(0.0, description="Сумма платежа в мае")
    PAY_AMT6: float = Field(0.0, description="Сумма платежа в апреле")

    class Config:
        json_schema_extra = {
            "example": {
                "LIMIT_BAL": 20000.0,
                "SEX": 2,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "AGE": 24,
                "PAY_0": 2,
                "BILL_AMT1": 3913.0,
                "PAY_AMT1": 0.0
            }
        }


class PredictionResponse(BaseModel):
    """Схема ответа с предсказанием"""
    default_prediction: int = Field(..., description="Предсказание дефолта (0 или 1)")
    default_probability: float = Field(..., description="Вероятность дефолта (0-1)")
    risk_level: str = Field(..., description="Уровень риска (Low/Medium/High)")


# Создание FastAPI приложения
app = FastAPI(
    title="Credit Default Prediction API",
    description="API для предсказания вероятности дефолта клиента по кредитной карте",
    version="1.0.0"
)

# Глобальная переменная для модели
model = None


def load_model():
    """Загрузка модели при старте приложения"""
    global model
    
    # Определение пути к модели
    model_path = os.getenv('MODEL_PATH', 'models/credit_default_model.pkl')
    
    # Если путь относительный, делаем его абсолютным
    if not os.path.isabs(model_path):
        # Попытка найти модель относительно корня проекта
        project_root = Path(__file__).resolve().parents[2]
        full_path = project_root / model_path
        
        # Если не найдено, проверяем другие варианты
        if not full_path.exists():
            # Проверяем gradient_boosting версию
            gb_path = project_root / 'models' / 'credit_default_model_gradient_boosting.pkl'
            if gb_path.exists():
                full_path = gb_path
            else:
                # Проверяем logistic_regression версию
                lr_path = project_root / 'models' / 'credit_default_model_logistic_regression.pkl'
                if lr_path.exists():
                    full_path = lr_path
        
        model_path = str(full_path)
    
    try:
        model = joblib.load(model_path)
        print(f"✅ Модель успешно загружена из {model_path}")
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")
        print(f"Путь к модели: {model_path}")
        raise


# Загрузка модели при старте приложения
@app.on_event("startup")
async def startup_event():
    """Событие при запуске приложения"""
    load_model()


@app.get("/")
async def read_root():
    """Корневой endpoint"""
    return {
        "message": "Credit Default Prediction API is alive!",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья API"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: ClientData):
    """
    Endpoint для предсказания вероятности дефолта
    
    Args:
        data: данные клиента
        
    Returns:
        Предсказание дефолта и вероятность
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Преобразование входных данных в DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # Выполнение предсказания
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0, 1]
        
        # Определение уровня риска
        if probability > 0.7:
            risk_level = "High"
        elif probability > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Формирование ответа
        response = PredictionResponse(
            default_prediction=int(prediction),
            default_probability=float(probability),
            risk_level=risk_level
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
