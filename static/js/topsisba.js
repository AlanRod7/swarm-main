document.addEventListener('DOMContentLoaded', function () {
    
    const ejecutarTopsisba = document.getElementById('ejecutarTopsisba');
    const formularioTopsisba = document.getElementById('topsisbaForm');
    

    ejecutarTopsisba.addEventListener('click', function () {
        console.log('ejecutarTopsisba clicked');

        //Obtener datos del formulario
        const formData = new FormData(formularioTopsisba);

        // Realizar la solicitud Ajax
        fetch('/topsisba', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())  // Parsea la respuesta como JSON
        .then(data => {
            console.log('Datos recibidos:', data);
            // Actualizar los campos de entrada con los nuevos datos
            const mejoresAlternativas = data.mejor_alternativa;

            for (let i = 0; i < mejoresAlternativas.length; i++) {
                document.getElementById(`alternativa${i}`).innerText = mejoresAlternativas[i];
                
            }

            document.getElementById('cantidadIteraciones').value = data.iteraciones;
            document.getElementById('horaInicio').value = data.hora_inicio;
            document.getElementById('fechaInicio').value = data.fecha_inicio;
            document.getElementById('horaFinalizacion').value = data.hora_finalizacion;
            document.getElementById('tiempoEjecucion').value = data.tiempo_ejecucion;
        })
        .catch(error => console.error('Error:', error));
    });


    

    // Evitar el envío tradicional del formulario
    formularioTopsisba.addEventListener('submit', function (event) {
        event.preventDefault();
    });

    
});
