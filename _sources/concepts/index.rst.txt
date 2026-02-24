Overview of the concepts
========================================
The implemented eigensolvers are based on a templated type `EVP` that must implement the Eigenproblem concept.
Each Eigenproblem must export certain types which themselves must adhere to certain rules that are also defined as concepts. This facilitates the development of new backends. The documentation of all the necessary concepts is given below. Detailed descriptions of the individual methods that these concepts require are given in the OpenMP backend documentation.

.. doxygenconcept:: trl::Eigenproblem
   :project: trl

.. doxygenconcept:: trl::BlockMultiVector
   :project: trl

.. doxygenconcept:: trl::BlockVectorView
   :project: trl

.. doxygenconcept:: trl::BlockMatrixConcept
   :project: trl

.. doxygenconcept:: trl::ReorthogonalizationStrategy
   :project: trl
