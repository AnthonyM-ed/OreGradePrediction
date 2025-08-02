## 0. Importante: Necesitas de una BD que cumpla con los requisitos, usar un .env para establecer el acceso a las tablas.
DB_USER=
DB_PASSWORD=
DB_SERVER=
DB_DATABASE=
DB_PORT=

### 1. Inicializar Backend (Django)
```bash
cd backend
pip install requirements.txt
python manage.py runserver
```

### 2. Inicializar Frontend (React)
```bash
cd client
npm run dev
```

### 3. Test
```bash
cd backend
python test_prediction_api.py
```

## URLs

- **Frontend Application**: http://localhost:5173
- **Backend API**: http://127.0.0.1:8000
