document.addEventListener('DOMContentLoaded', function () {
    const ejecutarDapso = document.getElementById('ejecutarDapso');
    const formularioDapso = document.getElementById('dapsoForm');
    
    ejecutarDapso.addEventListener('click', function () {
        console.log('Ejecutar DAPSO clicked');

        //Obtener datos del formulario
        const formData = new FormData(formularioDapso);

        // Realizar la solicitud Ajax
        fetch('/dapso', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())  // Parsea la respuesta como JSON
        .then(data => {
            console.log('Datos recibidos:', data);
            // Actualizar los campos de entrada con los nuevos datos
            const mejoresAlternativas = data.mejor_alternativa;

            for (let i = 0; i < mejoresAlternativas.length; i++) {
                document.getElementById(`alternativaDapso${i}`).innerText = mejoresAlternativas[i];
                
            }

            document.getElementById('iteracionesDapso').value = data.iteraciones;
            document.getElementById('horaInicioDapso').value = data.hora_inicio;
            document.getElementById('fechaInicioDapso').value = data.fecha_inicio;
            document.getElementById('horaFinalizacionDapso').value = data.hora_finalizacion;
            document.getElementById('tiempoEjecucionDapso').value = data.tiempo_ejecucion;
        })
        .catch(error => console.error('Error:', error));
    });

    // Evitar el envío tradicional del formulario
    formularioDapso.addEventListener('submit', function (event) {
        event.preventDefault();
    });
});
