

document.addEventListener('DOMContentLoaded', function () {
    const ejecutarComparacion = document.getElementById('ejecutarcomparacionAco');
    const formularioComparacion = document.getElementById('comparacionFormAco');
    ejecutarComparacion.addEventListener('click', function () {
        solicitudPso(formularioComparacion)
            .then(() => solicitudDapso(formularioComparacion))
            //.then(() => solicitudMoorapso(formularioComparacion))
            //.then(() => solicitudTopsispso(formularioComparacion))
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
        fetch('/aco', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())  // Parsea la respuesta como JSON
            .then(data => {
                // Actualizar los campos de entrada con los nuevos datos
                const mejoresAlternativas = data.mejor_alternativa;
                for (let i = 0; i < mejoresAlternativas.length; i++) {
                    document.getElementById(`alternativaAco${i}`).innerText = mejoresAlternativas[i];
                }
                document.getElementById('ejecucionAco').value = data.tiempo_ejecucion;
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
        fetch('/topsisba', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())  // Parsea la respuesta como JSON
            .then(data => {
                // Actualizar los campos de entrada con los nuevos datos
                const mejoresAlternativas = data.mejor_alternativa;
                for (let i = 0; i < mejoresAlternativas.length; i++) {
                    document.getElementById(`alternativaDaaco${i}`).innerText = mejoresAlternativas[i];
                }
                document.getElementById('ejecucionDaaco').value = data.tiempo_ejecucion;
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