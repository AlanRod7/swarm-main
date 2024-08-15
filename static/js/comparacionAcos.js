
document.addEventListener('DOMContentLoaded', function () {

    const ejecutarComparacion = document.getElementById('ejecutarComparacionAco');
    const formularioComparacion = document.getElementById('comparacionFormAco');
    ejecutarComparacion.addEventListener('click', function () {
        solicitudAco(formularioComparacion)
            .then(() => solicitudDaAco(formularioComparacion))
            .then(() => solicitudMooraAco(formularioComparacion))
            .then(() => solicitudTopsisAco(formularioComparacion))
    });
    // Evitar el envío tradicional del formulario
    formularioComparacion.addEventListener('submit', function (event) {
        event.preventDefault();
    });
});


//-------------------------Solicitudes------------------------

const solicitudAco = (formularioComparacion) => {
    //Obtener datos del formulario

    return new Promise((resolve, reject) => {
        // Realizar la solicitud Ajax
        const formData = new FormData(formularioComparacion);
        fetch('/aco', {
            method: 'POST',
            body: formData,
            headers: {
                'Cache-Control': 'no-cache'
            }
        })
            .then(response => response.json())  // Parsea la respuesta como JSON
            .then(data => {
                // Actualizar los campos de entrada con los nuevos datos
                const mejoresAlternativas = data.mejor_alternativa;
                for (let i = 0; i < mejoresAlternativas.length; i++) {
                    document.getElementById(`alternativaAco${i}`).innerText = '';  // Limpiar el texto
                    document.getElementById(`alternativaAco${i}`).innerText = mejoresAlternativas[i];
                }
                document.getElementById('ejecucionAco').value = data.tiempo_ejecucion;
                resolve();
            })
            .catch(error => { console.error('Error:', error); reject(error); });
    })
}

const solicitudDaAco = (formularioComparacion) => {

    return new Promise((resolve, reject) => {
        //Obtener datos del formulario
        const formData = new FormData(formularioComparacion);
        // Realizar la solicitud Ajax
        fetch('/daaco', {
            method: 'POST',
            body: formData,
            headers: {
                'Cache-Control': 'no-cache'
            }
        })
            .then(response => response.json())  // Parsea la respuesta como JSON
            .then(data => {
                if (data && Array.isArray(data.mejor_alternativa)) {
                    const mejoresAlternativas = data.mejor_alternativa;
                    for (let i = 0; i < mejoresAlternativas.length; i++) {
                        document.getElementById(`alternativaDaaco${i}`).innerText = mejoresAlternativas[i];
                    }
                } else {
                    console.error("Formato de datos incorrecto", data);
                }
                document.getElementById('ejecucionDaaco').value = data.tiempo_ejecucion;
                resolve();
            })
            .catch(error => { console.error('Error:', error); reject(error); });
    })
}
const solicitudMooraAco = (formularioComparacion) => {
    return new Promise((resolve, reject) => {
        //Obtener datos del formulario
        const formData = new FormData(formularioComparacion);
        // Realizar la solicitud Ajax
        fetch('/mooraaco', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())  // Parsea la respuesta como JSON
            .then(data => {
                // Actualizar los campos de entrada con los nuevos datos
                const mejoresAlternativas = data.mejor_alternativa;
                for (let i = 0; i < mejoresAlternativas.length; i++) {
                    document.getElementById(`alternativaMooraaco${i}`).innerText = mejoresAlternativas[i];
                }
                document.getElementById('ejecucionMooraaco').value = data.tiempo_ejecucion;
                resolve();
            })
            .catch(error => { console.error('Error:', error); reject(error); });
    })
}

const solicitudTopsisAco = (formularioComparacion) => {
    return new Promise((resolve, reject) => {
        //Obtener datos del formulario
        const formData = new FormData(formularioComparacion);
        // Realizar la solicitud Ajax
        fetch('/topsisaco', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())  // Parsea la respuesta como JSON
            .then(data => {
                // Actualizar los campos de entrada con los nuevos datos
                const mejoresAlternativas = data.mejor_alternativa;
                for (let i = 0; i < mejoresAlternativas.length; i++) {
                    document.getElementById(`alternativaTopsisaco${i}`).innerText = mejoresAlternativas[i];
                }
                document.getElementById('ejecucionTopsisaco').value = data.tiempo_ejecucion;
                resolve();
            })
            .catch(error => { console.error('Error:', error); reject(error); });
    })
}