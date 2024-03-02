document.addEventListener('DOMContentLoaded', function () {
    
    const ejecutarComparacion = document.getElementById('ejecutarComparacionPso');
    const formularioComparacion = document.getElementById('comparacionFormPso');
    
    ejecutarComparacion.addEventListener('click', function () {
        console.log('Ejecutar Comparacion clicked');

        solicitudPso(formularioComparacion);
        solicitudDapso(formularioComparacion);
        solicitudMoorapso(formularioComparacion);
        solicitudTopsispso(formularioComparacion);
    });

    // Evitar el envío tradicional del formulario
    formularioComparacion.addEventListener('submit', function (event) {
        event.preventDefault();
    });
});


//-------------------------Solicitudes------------------------

const solicitudPso = (formularioComparacion) => {

    //Obtener datos del formulario
    const formData = new FormData(formularioComparacion);

     // Realizar la solicitud Ajax
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
    })
    .catch(error => console.error('Error:', error));
}

const solicitudDapso = (formularioComparacion) => {
    //Obtener datos del formulario
    const formData = new FormData(formularioComparacion);

     // Realizar la solicitud Ajax
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
            document.getElementById(`alternativaDapso${i}`).innerText = mejoresAlternativas[i];   
        }

    })
    .catch(error => console.error('Error:', error));
}

const solicitudMoorapso = (formularioComparacion) => {
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

    })
    .catch(error => console.error('Error:', error));
}

const solicitudTopsispso = (formularioComparacion) => {
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

    })
    .catch(error => console.error('Error:', error));
}