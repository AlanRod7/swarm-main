

document.addEventListener('DOMContentLoaded', function () {

    const ejecutarComparacion = document.getElementById('ejecutarComparacionPso');
    const formularioComparacion = document.getElementById('comparacionFormPso');

    ejecutarComparacion.addEventListener('click', function () {
        console.log('Ejecutar Comparacion clicked');


        solicitudPso(formularioComparacion)
            .then(() => solicitudDapso(formularioComparacion))
            .then(() => solicitudMoorapso(formularioComparacion))
            .then(() => solicitudTopsispso(formularioComparacion))
    });

    // Evitar el envío tradicional del formulario
    formularioComparacion.addEventListener('submit', function (event) {
        event.preventDefault();
    });
});


//-------------------------Solicitudes------------------------

const solicitudPso = (formularioComparacion) => {
    //Obtener datos del formulario

    return new Promise((resolve, reject) => {
        // Realizar la solicitud Ajax
        const formData = new FormData(formularioComparacion);

        fetch('/pso', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())  // Parsea la respuesta como JSON
            .then(data => {
                console.log('Datos recibidos:', data);
                // Actualizar los campos de entrada con los nuevos datos
                const mejoresAlternativas = data.mejor_alternativa;

                for (let i = 0; i < mejoresAlternativas.length; i++) {
                    document.getElementById(`alternativaPso${i}`).innerText = mejoresAlternativas[i];
                }

                document.getElementById('ejecucionPso').value = data.tiempo_ejecucion;
                resolve();

            })
            .catch(error => { console.error('Error:', error); reject(error); });

    })
}

const solicitudDapso = (formularioComparacion) => {
    return new Promise((resolve, reject) => {
        //Obtener datos del formulario
        const formData = new FormData(formularioComparacion);

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
                document.getElementById('ejecucionDapso').value = data.tiempo_ejecucion;
                resolve();
            })
            .catch(error => { console.error('Error:', error); reject(error); });
    })
}
const solicitudMoorapso = (formularioComparacion) => {
    return new Promise((resolve, reject) => {
        //Obtener datos del formulario
        const formData = new FormData(formularioComparacion);

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
                document.getElementById('ejecucionMoorapso').value = data.tiempo_ejecucion;
                resolve();
            })
            .catch(error => { console.error('Error:', error); reject(error); });
    })
}

const solicitudTopsispso = (formularioComparacion) => {
    return new Promise((resolve, reject) => {
        //Obtener datos del formulario
        const formData = new FormData(formularioComparacion);

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
                    document.getElementById(`alternativaTopsispso${i}`).innerText = mejoresAlternativas[i];
                }
                document.getElementById('ejecucionTopsispso').value = data.tiempo_ejecucion;
                resolve();
            })
            .catch(error => { console.error('Error:', error); reject(error); });
    })
}