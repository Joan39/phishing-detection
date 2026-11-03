# QPhish frontend

Install dependencies and start the development server from the project root:

```powershell
npm --prefix frontend install
npm --prefix frontend run dev
```

The frontend uses `VITE_API_BASE_URL` when it is set. Otherwise it connects to
`http://localhost:8000`. Sign in to the API before scanning URLs or submitting
feedback.
