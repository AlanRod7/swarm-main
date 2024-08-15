

document.addEventListener('DOMContentLoaded', function () {

    const ejecutarComparacionGeneral = document.getElementById('ejecutarComparacionGeneral');
    const formularioComparacionGeneral = document.getElementById('comparacionGeneral');

    ejecutarComparacionGeneral.addEventListener('click', function () {
        solicitudDapso(formularioComparacionGeneral)
            .then(() => solicitudMoorapso(formularioComparacionGeneral))
            .then(() => solicitudTopsispso(formularioComparacionGeneral))
            .then(() => solicitudDaba(formularioComparacionGeneral))
            .then(() => solicitudMooraba(formularioComparacionGeneral))
            .then(() => solicitudTopsisba(formularioComparacionGeneral))
            .then(() => solicitudDaaco(formularioComparacionGeneral))
            .then(() => solicitudMooraaco(formularioComparacionGeneral))
            .then(() => solicitudTopsisaco(formularioComparacionGeneral))
    });

    // Evitar el envío tradicional del formulario
    formularioComparacionGeneral.addEventListener('submit', function (event) {
        event.preventDefault();
    });
});

//-------------------------Solicitudes PSO------------------------
const solicitudDapso = (formularioComparacionGeneral) => {
    return new Promise((resolve, reject) => {
        //Obtener datos del formulario
        const formData = new FormData(formularioComparacionGeneral);

        // Realizar la solicitud Ajax
        fetch('/dapso', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())  // Parsea la respuesta como JSON
            .then(data => {
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
const solicitudMoorapso = (formularioComparacionGeneral) => {
    return new Promise((resolve, reject) => {
        //Obtener datos del formulario
        const formData = new FormData(formularioComparacionGeneral);
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

//-------------------------Solicitudes BA------------------------
const solicitudDaba = (formularioComparacionGeneral) => {
    return new Promise((resolve, reject) => {
        //Obtener datos del formulario
        const formData = new FormData(formularioComparacionGeneral);
        // Realizar la solicitud Ajax
        fetch('/daba', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())  // Parsea la respuesta como JSON
            .then(data => {
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

const solicitudMooraba = (formularioComparacionGeneral) => {
    return new Promise((resolve, reject) => {
        //Obtener datos del formulario
        const formData = new FormData(formularioComparacionGeneral);

        // Realizar la solicitud Ajax
        fetch('/mooraba', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())  // Parsea la respuesta como JSON
            .then(data => {
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

const solicitudTopsisba = (formularioComparacionGeneral) => {
    return new Promise((resolve, reject) => {
        //Obtener datos del formulario
        const formData = new FormData(formularioComparacionGeneral);
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
                    document.getElementById(`alternativaTopsisba${i}`).innerText = mejoresAlternativas[i];
                }
                document.getElementById('ejecucionTopsisba').value = data.tiempo_ejecucion;
                resolve();
            })
            .catch(error => { console.error('Error:', error); reject(error); });
    })
}

//-------------------------Solicitudes ACO------------------------
const solicitudDaaco = (formularioComparacionGeneral) => {
    return new Promise((resolve, reject) => {
        //Obtener datos del formulario
        const formData = new FormData(formularioComparacionGeneral);
        // Realizar la solicitud Ajax
        fetch('/daaco', {
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

const solicitudMooraaco = (formularioComparacionGeneral) => {
    return new Promise((resolve, reject) => {
        //Obtener datos del formulario
        const formData = new FormData(formularioComparacionGeneral);
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

const solicitudTopsisaco = (formularioComparacionGeneral) => {
    return new Promise((resolve, reject) => {
        //Obtener datos del formulario
        const formData = new FormData(formularioComparacionGeneral);
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



