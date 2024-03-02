document.addEventListener('DOMContentLoaded', function () {
    
    const ejecutarComparacion = document.getElementById('ejecutarComparacionBa');
    const formularioComparacion = document.getElementById('comparacionFormBa');
    
    ejecutarComparacion.addEventListener('click', function () {
        console.log('Ejecutar Comparacion clicked');

        solicitudBa(formularioComparacion);
        solicitudDaba(formularioComparacion);
        solicitudMoorapso(formularioComparacion);
        solicitudTopsispso(formularioComparacion);
    });

    // Evitar el envío tradicional del formulario
    formularioComparacion.addEventListener('submit', function (event) {
        event.preventDefault();
    });
});


//-------------------------Solicitudes------------------------

const solicitudBa = (formularioComparacion) => {

    //Obtener datos del formulario
    const formData = new FormData(formularioComparacion);

     // Realizar la solicitud Ajax
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
    })
    .catch(error => console.error('Error:', error));
}

const solicitudDaba = (formularioComparacion) => {
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