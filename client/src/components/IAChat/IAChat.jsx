// src/components/IAChat.jsx
import { useState } from "react";

const IAChat = () => {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAsk = async () => {
    if (!question.trim()) return;

    setLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/api/ask/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      setResponse(data);
    } catch (err) {
      setResponse({ error: "Error al consultar la IA." });
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: "1rem", maxWidth: "600px", margin: "0 auto" }}>
      <h2>Asistente IA para Geología Minera</h2>
      <input
        type="text"
        placeholder="Ej: ¿Dónde hay más de 3 ppm de Au?"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        style={{ width: "100%", padding: "0.5rem", marginBottom: "0.5rem" }}
      />
      <button onClick={handleAsk} disabled={loading}>
        {loading ? "Consultando..." : "Preguntar"}
      </button>

      {response && (
        <div style={{ marginTop: "1rem" }}>
          {response.query && (
            <>
              <h4>Consulta SQL generada:</h4>
              <pre>{response.query}</pre>
            </>
          )}

          {response.answer && (
            <>
              <h4>Respuesta generada por IA:</h4>
              <p>{response.answer}</p>
            </>
          )}

          {response.result && (
            <>
              <h4>Resultados:</h4>
              <ul>
                {response.result.map((row, idx) => (
                  <li key={idx}>
                    {row.Hole_ID} - {row.Element} ({row.weighted_grade} g/t)
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}

    </div>
  );
};

export default IAChat;
