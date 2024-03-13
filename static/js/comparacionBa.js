

document.addEventListener('DOMContentLoaded', function () {

    const ejecutarComparacion = document.getElementById('ejecutarComparacionBa');
    const formularioComparacion = document.getElementById('comparacionFormBa');

    ejecutarComparacion.addEventListener('click', function () {
        console.log('Ejecutar Comparacion clicked');


        solicitudBa(formularioComparacion)
            .then(() => solicitudDaba(formularioComparacion))
            .then(() => solicitudMooraba(formularioComparacion))
            .then(() => solicitudTopsisba(formularioComparacion))
    });

    // Evitar el envío tradicional del formulario
    formularioComparacion.addEventListener('submit', function (event) {
        event.preventDefault();
    });
});


//-------------------------Solicitudes------------------------

const solicitudBa = (formularioComparacion) => {
    //Obtener datos del formulario

    return new Promise((resolve, reject) => {
        // Realizar la solicitud Ajax
        const formData = new FormData(formularioComparacion);

        fetch('/ba', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())  // Parsea la respuesta como JSON
            .then(data => {
                console.log('Datos recibidos:', data);
                // Actualizar los campos de entrada con los nuevos datos
                const mejoresAlternativas = data.mejor_alternativa;

                for (let i = 0; i < mejoresAlternativas.length; i++) {
                    document.getElementById(`alternativaBa${i}`).innerText = mejoresAlternativas[i];
                }

                document.getElementById('ejecucionBa').value = data.tiempo_ejecucion;
                resolve();

            })
            .catch(error => { console.error('Error:', error); reject(error); });

    })
}

const solicitudDaba = (formularioComparacion) => {
    return new Promise((resolve, reject) => {
        //Obtener datos del formulario
        const formData = new FormData(formularioComparacion);

        // Realizar la solicitud Ajax
        fetch('/daba', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())  // Parsea la respuesta como JSON
            .then(data => {
                console.log('Datos recibidos:', data);
                // Actualizar los campos de entrada con los nuevos datos
                const mejoresAlternativas = data.mejor_alternativa;

                for (let i = 0; i < mejoresAlternativas.length; i++) {
                    document.getElementById(`alternativaDaba${i}`).innerText = mejoresAlternativas[i];
                }
                document.getElementById('ejecucionDaba').value = data.tiempo_ejecucion;
                resolve();
            })
            .catch(error => { console.error('Error:', error); reject(error); });
    })
}
const solicitudMooraba = (formularioComparacion) => {
    return new Promise((resolve, reject) => {
        //Obtener datos del formulario
        const formData = new FormData(formularioComparacion);

        // Realizar la solicitud Ajax
        fetch('/mooraba', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())  // Parsea la respuesta como JSON
            .then(data => {
                console.log('Datos recibidos:', data);
                // Actualizar los campos de entrada con los nuevos datos
                const mejoresAlternativas = data.mejor_alternativa;

                for (let i = 0; i < mejoresAlternativas.length; i++) {
                    document.getElementById(`alternativaMooraba${i}`).innerText = mejoresAlternativas[i];
                }
                document.getElementById('ejecucionMooraba').value = data.tiempo_ejecucion;
                resolve();
            })
            .catch(error => { console.error('Error:', error); reject(error); });
    })
}

const solicitudTopsisba = (formularioComparacion) => {
    return new Promise((resolve, reject) => {
        //Obtener datos del formulario
        const formData = new FormData(formularioComparacion);

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
                    document.getElementById(`alternativaTopsisba${i}`).innerText = mejoresAlternativas[i];
                }
                document.getElementById('ejecucionTopsisba').value = data.tiempo_ejecucion;
                resolve();
            })
            .catch(error => { console.error('Error:', error); reject(error); });
    })
}