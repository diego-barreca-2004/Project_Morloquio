# Contribuire

1. Ogni dipendenza nuova deve avere una licenza open source compatibile: `make licenses` deve restare verde.
2. Ogni fonte dati nuova va in `configs/sources/` con licenza verificata e URL della licenza.
3. Test prima del codice; i test che usano la rete vanno marcati `@pytest.mark.live`.
4. Identificatori in inglese, commenti e documentazione in italiano.
5. Prima di una PR: `make check`.
