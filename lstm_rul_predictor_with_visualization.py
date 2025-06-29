# Импорт необходимых библиотек
import io  # Для работы с потоками ввода-вывода
import zipfile  # Для работы с zip-архивами
import requests  # Для HTTP-запросов
import numpy as np  # Для численных операций
import pandas as pd  # Для работы с табличными данными
import matplotlib.pyplot as plt  # Для визуализации
import seaborn as sns  # Для статистической визуализации
import torch  # Основной фреймворк для нейросетей
import torch.nn as nn  # Нейросетевые модули PyTorch
from torch.utils.data import TensorDataset, DataLoader  # Для работы с данными
from torchviz import make_dot  # Для визуализации графов вычислений
from sklearn.preprocessing import MinMaxScaler  # Для нормализации данных
from sklearn.model_selection import train_test_split  # Для разделения данных
import optuna  # Для оптимизации гиперпараметров
from optuna.visualization import plot_optimization_history, plot_param_importances  # Визуализация оптимизации
import csv  # для работы с CSV файлами (используется csv.QUOTE_NONE)

# -------------------------------
# 1. Data Loading and Preprocessing
# -------------------------------
class CMAPSSDataLoader:
    """Класс для загрузки и предварительной обработки данных CMAPSS (прогнозирование остаточного ресурса двигателей)"""
    
    def __init__(self, fd=1, sequence_length=30, rul_clip=125):
        """
        Инициализация загрузчика данных
        
        Параметры:
            fd (int): Номер набора данных (1-4)
            sequence_length (int): Длина временных последовательностей для LSTM
            rul_clip (int): Максимальное значение RUL (Remaining Useful Life)
        """
        self.fd = fd
        self.sequence_length = sequence_length
        self.rul_clip = rul_clip
        # Базовые колонки в исходных данных
        self.columns = ['unit_id', 'time'] + \
            [f'op_setting_{i}' for i in range(1,4)] + \
            [f'sensor_{i}' for i in range(1,22)]
        # Сенсоры с постоянными значениями, которые будут удалены
        self.to_drop = ['sensor_1','sensor_5','sensor_6','sensor_10',
                        'sensor_16','sensor_18','sensor_19']
        
    def download_and_extract(self):
        """Загрузка zip-архива с данными с Kaggle"""
        url = "https://www.kaggle.com/api/v1/datasets/download/behrad3d/nasa-cmaps"
        resp = requests.get(url, stream=True)
        resp.raise_for_status()  # Проверка на ошибки HTTP
        return io.BytesIO(resp.content)  # Возвращаем содержимое как байтовый поток
    
    def load_data_from_zip(self, zip_bytes):
        """Извлечение данных из zip-архива"""
        with zipfile.ZipFile(zip_bytes) as z:
            fl = z.namelist()  # Получаем список файлов в архиве
            # Формируем пути к файлам на основе номера набора данных
            train_f = f'CMaps/train_FD00{self.fd}.txt'
            test_f  = f'CMaps/test_FD00{self.fd}.txt'
            rul_f   = f'CMaps/RUL_FD00{self.fd}.txt'
            
            # Проверка наличия необходимых файлов
            if train_f not in fl or test_f not in fl or rul_f not in fl:
                raise FileNotFoundError("Missing files in zip")
            
            data = {}
            # Чтение train и test данных
            for name, fname in [('train', train_f), ('test', test_f)]:
                raw = z.read(fname).decode('utf-8')
                # Предварительная обработка текстовых данных
                raw = '\n'.join(' '.join(line.split()) 
                                for line in raw.splitlines() if line.strip())
                # Чтение данных в DataFrame
                data[name] = pd.read_csv(io.StringIO(raw),
                                         sep='\s+', header=None,
                                         engine='python', quoting=csv.QUOTE_NONE)
            # Чтение RUL данных
            data['RUL'] = pd.read_csv(io.StringIO(
                z.read(rul_f).decode('utf-8')),
                sep='\s+', header=None, names=['RUL'],
                engine='python', quoting=csv.QUOTE_NONE)
            return data
    
    def preprocess(self, data):
        """Предварительная обработка данных"""
        for part in ['train','test']:
            df = data[part]
            # Установка названий столбцов и удаление ненужных
            df.columns = self.columns
            df.drop(columns=self.to_drop, inplace=True)
            
            # Расчет RUL (остаточного ресурса)
            df['RUL'] = df.groupby('unit_id')['time'] \
                           .transform(lambda x: x.max() - x)
            # Ограничение максимального значения RUL для устойчивости модели
            df['RUL'] = df['RUL'].clip(upper=self.rul_clip)
            
            # Генерация новых признаков: разности и скользящие средние
            sensors = [c for c in df.columns if c.startswith('sensor')]
            for s in sensors:
                # Разность между текущим и предыдущим значением
                df[f'{s}_diff'] = df.groupby('unit_id')[s].diff().fillna(0)
                # Скользящее среднее по 3 точкам
                df[f'{s}_ma3']  = df.groupby('unit_id')[s] \
                                  .rolling(window=3).mean() \
                                  .reset_index(level=0, drop=True).fillna(df[s])
        return data
    
    def normalize(self, train, test):
        """Нормализация данных с использованием MinMaxScaler"""
        train_y = train['RUL'].values
        test_y  = test['RUL'].values
        # Выбор признаков для нормализации (исключаем идентификаторы и целевую переменную)
        feats = [c for c in train.columns 
                 if c not in ['unit_id','time','RUL']]
        
        # Инициализация и применение нормализации
        scaler = MinMaxScaler()
        train_f = scaler.fit_transform(train[feats])  # fit только на train
        test_f  = scaler.transform(test[feats])  # transform на test
        
        # Создание новых DataFrame с нормализованными данными
        train_df = pd.DataFrame(train_f, columns=feats, index=train.index)
        test_df  = pd.DataFrame(test_f,  columns=feats, index=test.index)
        
        # Восстановление столбцов unit_id и time
        train_df[['unit_id','time']] = train[['unit_id','time']]
        test_df [['unit_id','time']] = test [['unit_id','time']]
        
        # Восстановление RUL
        train_df['RUL'] = train_y
        test_df ['RUL'] = test_y
        return train_df, test_df, scaler
    
    def create_sequences(self, df):
        """Создание временных последовательностей для LSTM"""
        X, y = [], []
        seq = self.sequence_length
        # Выбор признаков (исключаем идентификаторы и целевую переменную)
        feats = [c for c in df.columns 
                 if c not in ['unit_id','time','RUL']]
        
        # Генерация последовательностей для каждого двигателя
        for uid, group in df.groupby('unit_id'):
            arr = group[feats].values
            labels = group['RUL'].values
            # Создание последовательностей заданной длины
            for i in range(len(arr) - seq):
                X.append(arr[i:i+seq])
                y.append(labels[i+seq-1])  # RUL в последний момент последовательности
        return np.array(X), np.array(y)
    
    def prepare(self):
        """Полный процесс подготовки данных"""
        # 1. Загрузка данных
        zipb = self.download_and_extract()
        raw  = self.load_data_from_zip(zipb)
        
        # 2. Предварительная обработка
        proc = self.preprocess(raw)
        
        # 3. Нормализация
        tr, te, scaler = self.normalize(proc['train'], proc['test'])
        
        # 4. Создание последовательностей
        Xtr, ytr = self.create_sequences(tr)
        Xte, yte = self.create_sequences(te)
        
        # 5. Разделение на train и validation
        Xtr, Xval, ytr, yval = train_test_split(
            Xtr, ytr, test_size=0.2, random_state=42
        )
        
        print("Shapes of datasets:",
              Xtr.shape, Xval.shape, Xte.shape)
        
        return {
            'X_train':Xtr, 'y_train':ytr,
            'X_val':Xval, 'y_val':yval,
            'X_test':Xte, 'y_test':yte,
            'scaler':scaler
        }

# -------------------------------
# 2. Model Architecture
# -------------------------------
class PredictiveMaintenanceModel(nn.Module):
    """Модель LSTM для прогнозирования остаточного ресурса"""
    
    def __init__(self, input_size, h1=128, h2=48, dropout=0.3):
        """
        Инициализация модели
        
        Параметры:
            input_size (int): Размерность входных данных
            h1 (int): Количество нейронов в первом LSTM слое
            h2 (int): Количество нейронов во втором LSTM слое
            dropout (float): Вероятность dropout
        """
        super().__init__()
        # Первый LSTM слой
        self.lstm1 = nn.LSTM(input_size, h1, batch_first=True)
        self.dr1   = nn.Dropout(dropout)  # Dropout после первого слоя
        
        # Второй LSTM слой
        self.lstm2 = nn.LSTM(h1, h2, batch_first=True)
        self.dr2   = nn.Dropout(dropout)  # Dropout после второго слоя
        
        # Полносвязный слой для выхода
        self.fc    = nn.Linear(h2, 1)
        
        # Инициализация весов
        for n, p in self.named_parameters():
            if 'weight' in n: 
                nn.init.xavier_uniform_(p)  # Инициализация Xavier для весов
            if 'bias' in n: 
                nn.init.constant_(p, 0.)  # Инициализация нулями для смещений
    
    def forward(self, x):
        """Прямой проход модели"""
        # Первый LSTM слой
        o, _ = self.lstm1(x)
        o = self.dr1(o)  # Применение dropout
        
        # Второй LSTM слой
        o, (h, _) = self.lstm2(o)
        o = self.dr2(o)  # Применение dropout
        
        # Полносвязный слой (берем только последний элемент последовательности)
        return self.fc(o[:, -1, :]).squeeze()

# -------------------------------
# 3. Training and Evaluation
# -------------------------------
class ModelTrainer:
    """Класс для обучения и оценки модели"""
    
    def __init__(self, data, patience=5):
        """
        Инициализация тренера
        
        Параметры:
            data (dict): Словарь с данными (train, val, test)
            patience (int): Количество эпох для ранней остановки
        """
        self.data = data
        # Определение устройства (GPU или CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.patience = patience

    def get_loaders(self, batch_size):
        """Создание DataLoader'ов для train, validation и test"""
        D = self.data
        
        def mk(ds_x, ds_y):
            """Создание TensorDataset из данных"""
            return TensorDataset(
                torch.FloatTensor(ds_x),  # Преобразование в тензор
                torch.FloatTensor(ds_y)
            )
        
        # Создание DataLoader'ов
        tr = DataLoader(mk(D['X_train'], D['y_train']), 
                       batch_size=batch_size, shuffle=True)
        vl = DataLoader(mk(D['X_val'], D['y_val']), 
                       batch_size=batch_size)
        te = DataLoader(mk(D['X_test'], D['y_test']), 
                       batch_size=batch_size)
        return tr, vl, te

    def train(self, model, loaders, epochs=50, lr=1e-3):
        """Обучение модели"""
        tr, vl, _ = loaders
        model.to(self.device)  # Перемещение модели на устройство
        
        # Функция потерь - Huber loss (устойчива к выбросам)
        criterion = nn.SmoothL1Loss()
        
        # Оптимизатор AdamW с L2 регуляризацией
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Планировщик скорости обучения
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=3)
        
        # Переменные для ранней остановки
        best, val_best = None, float('inf')
        wait = 0
        history = {'train': [], 'val': []}  # История потерь

        for ep in range(1, epochs+1):
            # Фаза обучения
            model.train()
            tot = 0
            for xb, yb in tr:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()  # Обнуление градиентов
                pred = model(xb)  # Прямой проход
                loss = criterion(pred, yb)  # Расчет потерь
                loss.backward()  # Обратное распространение
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Обрезка градиентов
                opt.step()  # Обновление весов
                tot += loss.item()  # Накопление потерь
            tr_loss = tot / len(tr)  # Средние потери на эпоху
            
            # Фаза валидации
            model.eval()
            tot = 0
            with torch.no_grad():
                for xb, yb in vl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    loss = criterion(model(xb), yb)
                    tot += loss.item()
            vl_loss = tot / len(vl)
            
            # Сохранение истории
            history['train'].append(tr_loss)
            history['val'].append(vl_loss)
            
            # Обновление планировщика
            sched.step(vl_loss)

            # Ранняя остановка
            if vl_loss < val_best:
                val_best = vl_loss
                best = model.state_dict()  # Сохранение лучших весов
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"Early stopping at epoch {ep}")
                    break

            # Логирование
            if ep % 5 == 0:
                lr_now = opt.param_groups[0]['lr']
                print(f"Epoch {ep} | Train {tr_loss:.2f} | Val {vl_loss:.2f} | LR {lr_now:.1e}")

        # Загрузка лучших весов
        model.load_state_dict(best)
        return model, history

    def evaluate(self, model, loader):
        """Оценка модели на тестовых данных"""
        model.eval()
        criterion = nn.SmoothL1Loss()
        preds, acts = [], []
        
        # Предсказание на тестовых данных
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                out = model(xb)
                preds.append(out.cpu().numpy())
                acts.append(yb.cpu().numpy())
        
        # Объединение результатов
        preds = np.concatenate(preds)
        acts = np.concatenate(acts)
        
        # Расчет метрик
        mse = ((preds - acts)**2).mean()  # Среднеквадратичная ошибка
        mae = np.abs(preds - acts).mean()  # Средняя абсолютная ошибка
        from sklearn.metrics import r2_score
        r2 = r2_score(acts, preds)  # Коэффициент детерминации
        
        print(f"Test MSE={mse:.2f} MAE={mae:.2f} R2={r2:.2f}")
        
        # Визуализация результатов
        plt.figure(figsize=(6, 6))
        plt.scatter(acts, preds, alpha=0.5)  # Фактические vs предсказанные
        plt.plot([acts.min(), acts.max()], [acts.min(), acts.max()], 'r--')  # Идеальная линия
        plt.xlabel("Actual RUL")
        plt.ylabel("Predicted RUL")
        plt.show()
        
        # Распределение ошибок
        sns.histplot(acts - preds, kde=True)
        plt.title("Error distribution")
        plt.show()

# -------------------------------
# 4. Hyperparameter Optimization
# -------------------------------
def objective(trial, data):
    """Функция для оптимизации гиперпараметров с помощью Optuna"""
    # Подбор гиперпараметров
    params = {
        'h1': trial.suggest_int('h1', 64, 128, step=32),  # Размер первого LSTM слоя
        'h2': trial.suggest_int('h2', 32, 64, step=16),  # Размер второго LSTM слоя
        'drop': trial.suggest_float('drop', 0.1, 0.4),  # Dropout rate
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),  # Скорость обучения
        'bs': trial.suggest_categorical('bs', [32, 64])  # Размер батча
    }
    
    # Инициализация тренера и загрузчиков данных
    loader = ModelTrainer(data)
    tr, vl, _ = loader.get_loaders(params['bs'])
    
    # Создание модели с подобранными параметрами
    model = PredictiveMaintenanceModel(
        input_size=data['X_train'].shape[2],
        h1=params['h1'], h2=params['h2'],
        dropout=params['drop']
    )
    
    # Краткое обучение (для оптимизации)
    model, _ = loader.train(model, (tr, vl, None), epochs=5, lr=params['lr'])
    
    # Вычисление потерь на валидации
    model.eval()
    tot = 0
    crit = nn.SmoothL1Loss()
    with torch.no_grad():
        for xb, yb in vl:
            xb, yb = xb.to(loader.device), yb.to(loader.device)
            tot += crit(model(xb), yb).item()
    
    return tot / len(vl)  # Возвращаем средние потери для оптимизации

def optimize(data):
    """Оптимизация гиперпараметров с помощью Optuna"""
    study = optuna.create_study(direction='minimize')  # Минимизация потерь
    study.optimize(lambda t: objective(t, data), n_trials=10)  # 10 испытаний
    
    print("Best parameters:", study.best_trial.params)
    
    # Визуализация результатов оптимизации
    plot_optimization_history(study)
    plot_param_importances(study)
    
    return study.best_trial.params

# -------------------------------
# 5. Main
# -------------------------------
def main():
    """Основная функция"""
    # 1. Подготовка данных
    dl = CMAPSSDataLoader(fd=1, sequence_length=30, rul_clip=125)
    data = dl.prepare()

    # 2. Оптимизация гиперпараметров
    print("Optimizing hyperparameters...")
    best = optimize(data)

    # 3. Инициализация тренера и загрузчиков данных
    trainer = ModelTrainer(data, patience=7)
    tr, vl, te = trainer.get_loaders(best['bs'])
    
    # 4. Создание модели с лучшими параметрами
    model = PredictiveMaintenanceModel(
        input_size=data['X_train'].shape[2],
        h1=best['h1'], h2=best['h2'],
        dropout=best['drop']
    )
    
    # 5. Обучение модели
    print("Training final model...")
    model, hist = trainer.train(model, (tr, vl, None),
                               epochs=50, lr=best['lr'])
    
    # 6. Оценка на тестовых данных
    print("Evaluating on test set...")
    trainer.evaluate(model, te)

if __name__ == "__main__":
    main()