document.addEventListener('DOMContentLoaded', function () {
    const ejecutarMoorapso = document.getElementById('ejecutarMoorapso');
    const formularioMoorapso = document.getElementById('moorapsoForm');
    
    ejecutarMoorapso.addEventListener('click', function () {
        console.log('Ejecutar MOORA - PSO clicked');

        //Obtener datos del formulario
        const formData = new FormData(formularioMoorapso);

        // Realizar la solicitud Ajax
        fetch('/moorapso', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())  // Parsea la respuesta como JSON
        .then(data => {
            console.log('Datos recibidos:', data);
            // Actualizar los campos de entrada con los nuevos datos
            const mejoresAlternativas = data.mejor_alternativa;

            for (let i = 0; i < mejoresAlternativas.length; i++) {
                document.getElementById(`alternativaMoorapso${i}`).innerText = mejoresAlternativas[i];
                
            }

            document.getElementById('iteracionesMoorapso').value = data.iteraciones;
            document.getElementById('horaInicioMoorapso').value = data.hora_inicio;
            document.getElementById('fechaInicioMoorapso').value = data.fecha_inicio;
            document.getElementById('horaFinalizacionMoorapso').value = data.hora_finalizacion;
            document.getElementById('tiempoEjecucionMoorapso').value = data.tiempo_ejecucion;
        })
        .catch(error => console.error('Error:', error));
    });

    // Evitar el envío tradicional del formulario
    formularioMoorapso.addEventListener('submit', function (event) {
        event.preventDefault();
    });
});
