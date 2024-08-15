

document.addEventListener('DOMContentLoaded', function () {
    const ejecutarComparacion = document.getElementById('ejecutarComparacionAco');
    const formularioComparacion = document.getElementById('comparacionFormAco');
    ejecutarComparacion.addEventListener('click', async function () {
        await solicitudAco(formularioComparacion)
        await solicitudDaAco(formularioComparacion)
        await solicitudMooraAco(formularioComparacion)
        await solicitudTopsisAco(formularioComparacion)
    });
    // Evitar el envío tradicional del formulario
    formularioComparacion.addEventListener('submit', function (event) {
        event.preventDefault();
    });
});

//-------------------------Solicitudes------------------------
const solicitudAco = async (formularioComparacion) => {
    await delay(1000);  // Añade un retraso de 1 segundo

    // Obtener datos del formulario
    const formData = new FormData(formularioComparacion);

    try {
        const response = await fetch('/aco', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        // Actualizar los campos de entrada con los nuevos datos
        const mejoresAlternativas = data.mejor_alternativa;
        console.log(mejoresAlternativas)
        for (let i = 0; i < mejoresAlternativas.length; i++) {
            document.getElementById(`alternativaAco${i}`).innerText = mejoresAlternativas[i];
        }
        document.getElementById('ejecucionAco').value = data.tiempo_ejecucion;
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}

const solicitudDaAco = async (formularioComparacion) => {
    await delay(7000);

    const formData = new FormData(formularioComparacion);

    try {
        const response = await fetch('/daaco', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        const mejoresAlternativa = data.mejor_alternativa;
        console.log(mejoresAlternativa)
        for (let i = 0; i < mejoresAlternativa.length; i++) {
            document.getElementById(`alternativaDaaco${i}`).innerText = mejoresAlternativa[i];
        }
        document.getElementById('ejecucionDaaco').value = data.tiempo_ejecucion;
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}
const solicitudMooraAco = async (formularioComparacion) => {
    await delay(1000);

    const formData = new FormData(formularioComparacion);

    try {
        const response = await fetch('/mooraaco', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        const mejoresAlternativas = data.mejor_alternativa;
        for (let i = 0; i < mejoresAlternativas.length; i++) {
            document.getElementById(`alternativaMooraaco${i}`).innerText = mejoresAlternativas[i];
        }
        document.getElementById('ejecucionMooraaco').value = data.tiempo_ejecucion;
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}

const solicitudTopsisAco = async (formularioComparacion) => {
    await delay(1000);

    const formData = new FormData(formularioComparacion);

    try {
        const response = await fetch('/topsisaco', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        const mejoresAlternativas = data.mejor_alternativa;
        for (let i = 0; i < mejoresAlternativas.length; i++) {
            document.getElementById(`alternativaTopsisaco${i}`).innerText = mejoresAlternativas[i];
        }
        document.getElementById('ejecucionTopsisaco').value = data.tiempo_ejecucion;
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}