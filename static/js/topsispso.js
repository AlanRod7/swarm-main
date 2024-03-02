document.addEventListener('DOMContentLoaded', function () {
    
    const ejecutarTopsispso = document.getElementById('ejecutarTopsispso');
    const topsispsoForm = document.getElementById('topsispsoForm');
    

    ejecutarTopsispso.addEventListener('click', function () {
        console.log('ejecutarBtn clicked');

        //Obtener datos del formulario
        const formData = new FormData(topsispsoForm);

        // Realizar la solicitud Ajax
        fetch('/topsispso', {
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


    ejecutarDapso.addEventListener('click', function () {
        console.log('Ejecutar TOPSISPSO clicked');

        //Obtener datos del formulario
        const formData = new FormData(formulario);

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
    formulario.addEventListener('submit', function (event) {
        event.preventDefault();
    });

    // Evitar el envío tradicional del formulario
    formularioDapso.addEventListener('submit', function (event) {
        event.preventDefault();
    });
});
